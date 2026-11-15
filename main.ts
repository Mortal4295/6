// free-model-aggregator-01: one server, two client wires, ten simulated free providers.
//
// The trial hands this program a brief (MAKEFUN_PROMPT_FILE) with three documents: the
// provider directory, the quality table (ratings + capabilities + free flags), and the
// workload manifest (the 20 requests, in arrival order, with what each one requires).
// The env (MAKEFUN_ENV_URL) serves the providers under /p/<id>, plus /register and /work.
//
// Flow: probe every provider's catalog route in its own wire, build one catalog, listen
// on a free port, register {protocol_version: 1, port, catalog}, then serve the client's
// POST /v1/chat/completions and POST /v1/messages until GET /work says done, and exit 0.
// `node main.ts --list` prints the same catalog document and exits before serving.
//
// Routing: for each request, rank every free model that meets its requirements by the
// quality table's rating_milli (best first, provider/model id as the deterministic
// tie-break), try them in order, and fail over on refusal. A refusal whose message says
// the free-tier budget is spent (or the provider is out) marks that provider exhausted
// for the trial; any other refusal (a scheduled fail_call, a 5xx, a network error) is
// transient: retry the same provider once, then move to the next candidate, leaving the
// provider available for later requests. Exactly one successful provider call is made
// per request.
//
// Translation is exact, never approximate: the provider answers with a canned text
// `ANSWER ECHO{...}` whose fields (system hash, user hash, tools, max_tokens, stops,
// image hash, json mode) are computed from the request as the client sent it, so the
// system prompt, user text, image bytes, tool names, max_tokens, stop sequences and
// JSON mode must all survive the wire translation byte for byte. Usage tokens and the
// stop family are carried through the same way, in whatever wire the client used.
//
// Everything here is deterministic: no wall clock, no unseeded randomness, fixed
// `created` values in listings, so a replay reproduces the same bytes.

import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const stringOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numberOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);
const numberOrZero = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

/** A string or an array of strings, as a list. */
function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The text of a string, or of a block array's text blocks joined with newlines. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest
// ---------------------------------------------------------------------------

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: WorkloadRequest[];
  error: string | null;
}

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  const base =
    stringOf(v.path) !== ''
      ? stringOf(v.path)
      : stringOf(v.base_path) !== ''
        ? stringOf(v.base_path)
        : stringOf(v.base_url) !== ''
          ? stringOf(v.base_url)
          : stringOf(v.base);
  return {
    id,
    name: stringOf(v.name) !== '' ? stringOf(v.name) : id,
    wire,
    auth,
    base,
    key_env: stringOf(v.key_env),
    key: stringOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = workloadRequestOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, manifest, error: null };
}

function ratingRow(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

// ---------------------------------------------------------------------------
// Catalog probe
// ---------------------------------------------------------------------------

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** One model as a provider's catalog route listed it; null members are what it did not say. */
interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** A price field of a listing: true when every price is zero, null when it says nothing. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context =
    numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name:
      stringOf(v.display_name) !== ''
        ? stringOf(v.display_name)
        : stringOf(v.displayName) !== ''
          ? stringOf(v.displayName)
          : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok)
      return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

/** Whether a model is free: the quality table's row where it has one (the sealed truth), else the listing. */
function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** What a model can do: the quality table's row where it has one, the listing's fields where it does not. */
function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (
    requires.min_context !== undefined &&
    (caps.context_length === null || caps.context_length < requires.min_context)
  )
    return false;
  return true;
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The catalog the probes and the quality table make: every provider that answered, every
 * model it listed. Its base_url is the directory's own path (/p/<id>), never the path
 * joined to the env URL: the env's origin differs between a live pass and its replay,
 * and the catalog is part of the state export a replay must reproduce byte for byte.
 */
function buildCatalog(probes: readonly Probe[], ratings: RatingRow[]): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRow(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: modelFree(model, row),
        context_length: row?.context_length ?? model.context_length ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// The normalized request: one shape for both client wires, three upstream wires
// ---------------------------------------------------------------------------

interface ImageRef {
  mime: string;
  data: string;
}
type ImageSource = { kind: 'base64'; mime: string; data: string } | { kind: 'url'; url: string };

interface ToolCall {
  id: string;
  name: string;
  args: string; // JSON string of the arguments object
}

interface Turn {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  image: ImageSource | null;
  toolCalls: ToolCall[];
  toolCallId: string;
}

interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

interface ToolChoice {
  kind: 'auto' | 'any' | 'tool';
  name: string;
}

interface NormReq {
  systemText: string;
  turns: Turn[];
  tools: ToolDef[];
  toolChoice: ToolChoice | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  stream: boolean;
  jsonMode: boolean;
}

/** Split an image_url value into base64 data or a plain URL. */
function imageSourceOf(url: string): ImageSource | null {
  if (url === '') return null;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { kind: 'base64', mime: data[1] !== '' ? data[1] : 'image/png', data: data[2] };
  return { kind: 'url', url };
}

/** An OpenAI message's image part, if it has one. */
function openAIImageOf(content: unknown): ImageSource | null {
  if (!Array.isArray(content)) return null;
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
    const src = imageSourceOf(stringOf(block.image_url.url));
    if (src !== null) return src;
  }
  return null;
}

/** An Anthropic message's image block, if it has one. */
function anthropicImageOf(content: unknown): ImageSource | null {
  if (!Array.isArray(content)) return null;
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { kind: 'base64', mime: stringOf(source.media_type) !== '' ? stringOf(source.media_type) : 'image/png', data: source.data };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { kind: 'url', url: source.url };
  }
  return null;
}

/** A tool call's arguments as a JSON string. */
function argsJsonOf(value: unknown): string {
  if (typeof value === 'string') return value;
  if (isRecord(value)) return JSON.stringify(value);
  return '{}';
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argsObjectOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** An OpenAI Chat Completions body as a normalized request. */
function fromOpenAI(body: Json): NormReq {
  const systemParts: string[] = [];
  const turns: Turn[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const toolCalls: ToolCall[] = [];
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          toolCalls.push({
            id: stringOf(call.id),
            name: stringOf(call.function.name),
            args: argsJsonOf(call.function.arguments),
          });
        }
      }
      turns.push({ role: 'assistant', text: textOf(raw.content), image: null, toolCalls, toolCallId: '' });
      continue;
    }
    if (role === 'tool') {
      turns.push({
        role: 'tool',
        text: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
        image: null,
        toolCalls: [],
        toolCallId: stringOf(raw.tool_call_id),
      });
      continue;
    }
    // Any other role is a user turn.
    turns.push({ role: 'user', text: textOf(raw.content), image: openAIImageOf(raw.content), toolCalls: [], toolCallId: '' });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: stringOf(raw.function.description),
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  let toolChoice: ToolChoice | null = null;
  const choice = body.tool_choice;
  if (choice === 'auto') toolChoice = { kind: 'auto', name: '' };
  else if (choice === 'required' || choice === 'any') toolChoice = { kind: 'any', name: '' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    toolChoice = { kind: 'tool', name: stringOf(choice.function.name) };
  }
  return {
    systemText: systemParts.join('\n'),
    turns,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops: stringList(body.stop),
    stream: body.stream === true,
    jsonMode: format === 'json_object' || format === 'json_schema',
  };
}

/** An Anthropic Messages body as a normalized request. */
function fromAnthropic(body: Json): NormReq {
  const turns: Turn[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const toolCalls: ToolCall[] = [];
      if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          toolCalls.push({
            id: stringOf(block.id),
            name: stringOf(block.name),
            args: JSON.stringify(isRecord(block.input) ? block.input : {}),
          });
        }
      }
      turns.push({ role: 'assistant', text: textOf(content), image: null, toolCalls, toolCallId: '' });
      continue;
    }
    // User turns; a tool_result block becomes a tool turn of its own.
    if (Array.isArray(content)) {
      const parts: string[] = [];
      let image: ImageSource | null = null;
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') {
          const text =
            typeof block.content === 'string'
              ? block.content
              : Array.isArray(block.content) && textOf(block.content) !== ''
                ? textOf(block.content)
                : block.content === undefined
                  ? ''
                  : JSON.stringify(block.content);
          turns.push({ role: 'tool', text, image: null, toolCalls: [], toolCallId: stringOf(block.tool_use_id) });
          continue;
        }
        if (block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
        if (block.type === 'image' && image === null) image = anthropicImageOf([block]);
      }
      if (parts.length > 0 || image !== null) {
        turns.push({ role: 'user', text: parts.join('\n'), image, toolCalls: [], toolCallId: '' });
      }
      continue;
    }
    turns.push({ role: 'user', text: typeof content === 'string' ? content : '', image: null, toolCalls: [], toolCallId: '' });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: stringOf(raw.description),
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice | null = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = { kind: 'auto', name: '' };
    else if (choice.type === 'any') toolChoice = { kind: 'any', name: '' };
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { kind: 'tool', name: choice.name };
  }
  return {
    systemText: textOf(body.system),
    turns,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops: stringList(body.stop_sequences),
    stream: body.stream === true,
    jsonMode: false,
  };
}

// ---------------------------------------------------------------------------
// Upstream request builders: one normalized request, three wires
// ---------------------------------------------------------------------------

/** One normalized turn as OpenAI content parts. */
function openAIPartsOf(turn: Turn): Json[] {
  const parts: Json[] = [];
  if (turn.text !== '') parts.push({ type: 'text', text: turn.text });
  if (turn.image !== null) {
    if (turn.image.kind === 'base64') {
      parts.push({ type: 'image_url', image_url: { url: `data:${turn.image.mime};base64,${turn.image.data}` } });
    } else {
      parts.push({ type: 'image_url', image_url: { url: turn.image.url } });
    }
  }
  return parts;
}

/** The normalized request as an OpenAI Chat Completions body naming `model`. */
function toOpenAI(norm: NormReq, model: string): Json {
  const messages: Json[] = [];
  if (norm.systemText !== '') messages.push({ role: 'system', content: norm.systemText });
  for (const turn of norm.turns) {
    if (turn.role === 'assistant') {
      const message: Json = { role: 'assistant', content: turn.text !== '' ? turn.text : null };
      if (turn.toolCalls.length > 0) {
        message.tool_calls = turn.toolCalls.map((c) => ({
          id: c.id,
          type: 'function',
          function: { name: c.name, arguments: c.args },
        }));
      }
      messages.push(message);
      continue;
    }
    if (turn.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: turn.toolCallId, content: turn.text });
      continue;
    }
    const parts = openAIPartsOf(turn);
    messages.push({ role: 'user', content: parts.length === 1 && turn.image === null ? turn.text : parts });
  }
  const out: Json = { model, messages };
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop = norm.stops;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
  }
  if (norm.toolChoice !== null) {
    if (norm.toolChoice.kind === 'auto') out.tool_choice = 'auto';
    else if (norm.toolChoice.kind === 'any') out.tool_choice = 'required';
    else out.tool_choice = { type: 'function', function: { name: norm.toolChoice.name } };
  }
  if (norm.jsonMode) out.response_format = { type: 'json_object' };
  if (norm.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One normalized turn as Anthropic content blocks. */
function anthropicBlocksOf(turn: Turn): Json[] {
  const blocks: Json[] = [];
  if (turn.text !== '') blocks.push({ type: 'text', text: turn.text });
  if (turn.image !== null) {
    if (turn.image.kind === 'base64') {
      blocks.push({
        type: 'image',
        source: { type: 'base64', media_type: turn.image.mime, data: turn.image.data },
      });
    } else {
      blocks.push({ type: 'image', source: { type: 'url', url: turn.image.url } });
    }
  }
  return blocks;
}

/** The normalized request as an Anthropic Messages body naming `model`. */
function toAnthropic(norm: NormReq, model: string): Json {
  const messages: Json[] = [];
  for (const turn of norm.turns) {
    if (turn.role === 'assistant') {
      const content: Json[] = [];
      if (turn.text !== '') content.push({ type: 'text', text: turn.text });
      for (const call of turn.toolCalls) {
        content.push({ type: 'tool_use', id: call.id, name: call.name, input: argsObjectOf(call.args) });
      }
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (turn.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: turn.toolCallId, content: turn.text }],
      });
      continue;
    }
    const blocks = anthropicBlocksOf(turn);
    messages.push({ role: 'user', content: blocks.length === 1 && turn.image === null ? turn.text : blocks });
  }
  const out: Json = { model, messages };
  if (norm.systemText !== '') out.system = norm.systemText;
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop_sequences = norm.stops;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (norm.toolChoice !== null) {
    if (norm.toolChoice.kind === 'auto') out.tool_choice = { type: 'auto' };
    else if (norm.toolChoice.kind === 'any') out.tool_choice = { type: 'any' };
    else out.tool_choice = { type: 'tool', name: norm.toolChoice.name };
  }
  if (norm.stream) out.stream = true;
  return out;
}

/** One normalized turn as Gemini content parts. */
function geminiPartsOf(turn: Turn): Json[] {
  const parts: Json[] = [];
  if (turn.text !== '') parts.push({ text: turn.text });
  if (turn.image !== null) {
    if (turn.image.kind === 'base64') {
      parts.push({ inlineData: { mimeType: turn.image.mime, data: turn.image.data } });
    } else {
      parts.push({ fileData: { fileUri: turn.image.url } });
    }
  }
  for (const call of turn.toolCalls) {
    parts.push({ functionCall: { name: call.name, args: argsObjectOf(call.args) } });
  }
  return parts;
}

/** The normalized request as a Gemini generateContent body. */
function toGemini(norm: NormReq): Json {
  const contents: Json[] = [];
  for (const turn of norm.turns) {
    if (turn.role === 'assistant') {
      contents.push({ role: 'model', parts: geminiPartsOf(turn) });
      continue;
    }
    if (turn.role === 'tool') {
      contents.push({
        role: 'function',
        parts: [{ functionResponse: { name: '', response: { content: turn.text } } }],
      });
      continue;
    }
    contents.push({ role: 'user', parts: geminiPartsOf(turn) });
  }
  const out: Json = { contents };
  if (norm.systemText !== '') out.systemInstruction = { parts: [{ text: norm.systemText }] };
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.topP !== null) config.topP = norm.topP;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (norm.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: norm.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        })),
      },
    ];
  }
  return out;
}

// ---------------------------------------------------------------------------
// The normalized answer: one shape for all three provider wires
// ---------------------------------------------------------------------------

interface NormAns {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The provider's stop word: "stop", "end_turn", "STOP", "length", ... */
  stop: string;
  toolCalls: ToolCall[];
}

/** The text of an OpenAI message: a string, or its text parts joined. */
function openAITextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** SSE text as frames: {event, data} pairs, in order. */
function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonLine(payload: string): Json | null {
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI Chat Completions answer (JSON or SSE) as a normalized answer. */
function parseOpenAIAnswer(text: string, stream: boolean): NormAns | null {
  if (!stream) {
    const body = parseJsonLine(text);
    if (body === null) return null;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const toolCalls: ToolCall[] = [];
    if (Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: stringOf(call.id),
          name: stringOf(call.function.name),
          args: argsJsonOf(call.function.arguments),
        });
      }
    }
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: openAITextOf(message.content),
      inputTokens: numberOrZero(usage.prompt_tokens),
      outputTokens: numberOrZero(usage.completion_tokens),
      stop: stringOf(choice.finish_reason) !== '' ? stringOf(choice.finish_reason) : 'stop',
      toolCalls,
    };
  }
  let outText = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'stop';
  const toolParts = new Map<number, ToolCall>();
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    const chunk = parseJsonLine(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') outText += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (!isRecord(call)) continue;
        const index = typeof call.index === 'number' ? call.index : toolParts.size;
        const prev = toolParts.get(index) ?? { id: '', name: '', args: '' };
        const fn = isRecord(call.function) ? call.function : {};
        toolParts.set(index, {
          id: stringOf(call.id) !== '' ? stringOf(call.id) : prev.id,
          name: stringOf(fn.name) !== '' ? stringOf(fn.name) : prev.name,
          args: prev.args + (typeof fn.arguments === 'string' ? fn.arguments : ''),
        });
      }
    }
    if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') stop = choice.finish_reason;
  }
  return {
    text: outText,
    inputTokens,
    outputTokens,
    stop,
    toolCalls: [...toolParts.entries()].sort((a, b) => a[0] - b[0]).map(([, c]) => c),
  };
}

/** An Anthropic Messages answer (JSON or SSE) as a normalized answer. */
function parseAnthropicAnswer(text: string, stream: boolean): NormAns | null {
  if (!stream) {
    const body = parseJsonLine(text);
    if (body === null) return null;
    const toolCalls: ToolCall[] = [];
    let outText = '';
    if (Array.isArray(body.content)) {
      for (const block of body.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') outText += block.text;
        if (block.type === 'tool_use') {
          toolCalls.push({
            id: stringOf(block.id),
            name: stringOf(block.name),
            args: JSON.stringify(isRecord(block.input) ? block.input : {}),
          });
        }
      }
    }
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: outText,
      inputTokens: numberOrZero(usage.input_tokens),
      outputTokens: numberOrZero(usage.output_tokens),
      stop: stringOf(body.stop_reason) !== '' ? stringOf(body.stop_reason) : 'end_turn',
      toolCalls,
    };
  }
  let outText = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'end_turn';
  const toolParts = new Map<number, ToolCall>();
  for (const frame of parseSseFrames(text)) {
    const data = parseJsonLine(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_start' && isRecord(data.content_block)) {
      const index = numberOrZero(data.index);
      const block = data.content_block;
      if (block.type === 'tool_use') {
        toolParts.set(index, {
          id: stringOf(block.id),
          name: stringOf(block.name),
          args: '',
        });
      }
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const index = numberOrZero(data.index);
      const delta = data.delta;
      if (delta.type === 'text_delta' && typeof delta.text === 'string') outText += delta.text;
      if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
        const prev = toolParts.get(index);
        if (prev !== undefined) toolParts.set(index, { ...prev, args: prev.args + delta.partial_json });
      }
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string' && data.delta.stop_reason !== '') {
        stop = data.delta.stop_reason;
      }
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  return {
    text: outText,
    inputTokens,
    outputTokens,
    stop,
    toolCalls: [...toolParts.entries()].sort((a, b) => a[0] - b[0]).map(([, c]) => c),
  };
}

/** A Gemini generateContent answer (JSON or SSE) as a normalized answer. */
function parseGeminiAnswer(text: string, stream: boolean): NormAns | null {
  if (!stream) {
    const body = parseJsonLine(text);
    if (body === null) return null;
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    let outText = '';
    if (Array.isArray(content.parts)) {
      for (const part of content.parts) {
        if (isRecord(part) && typeof part.text === 'string') outText += part.text;
      }
    }
    const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
    return {
      text: outText,
      inputTokens: numberOrZero(usage.promptTokenCount),
      outputTokens: numberOrZero(usage.candidatesTokenCount),
      stop: stringOf(candidate.finishReason) !== '' ? stringOf(candidate.finishReason) : 'STOP',
      toolCalls: [],
    };
  }
  let outText = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'STOP';
  for (const frame of parseSseFrames(text)) {
    const chunk = parseJsonLine(frame.data);
    if (chunk === null) continue;
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      if (Array.isArray(content.parts)) {
        for (const part of content.parts) {
          if (isRecord(part) && typeof part.text === 'string') outText += part.text;
        }
      }
      if (typeof candidate.finishReason === 'string' && candidate.finishReason !== '') stop = candidate.finishReason;
    }
    if (isRecord(chunk.usageMetadata)) {
      inputTokens = numberOrZero(chunk.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
    }
  }
  return { text: outText, inputTokens, outputTokens, stop, toolCalls: [] };
}

/** Parse a provider's answer in its own wire. */
function parseProviderAnswer(wire: Wire, text: string, stream: boolean): NormAns | null {
  if (wire === 'openai') return parseOpenAIAnswer(text, stream);
  if (wire === 'anthropic') return parseAnthropicAnswer(text, stream);
  return parseGeminiAnswer(text, stream);
}

// ---------------------------------------------------------------------------
// Rendering: a normalized answer in the client's wire
// ---------------------------------------------------------------------------

/** A fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

/** A provider stop word as an OpenAI finish_reason. */
function openAIStopOf(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A provider stop word as an Anthropic stop_reason. */
function anthropicStopOf(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** A normalized answer as an OpenAI Chat Completions object naming `model`. */
function renderOpenAIJson(ans: NormAns, model: string): Json {
  const message: Json = { role: 'assistant', content: ans.text };
  if (ans.toolCalls.length > 0) {
    message.tool_calls = ans.toolCalls.map((c) => ({
      id: c.id !== '' ? c.id : 'call_unknown',
      type: 'function',
      function: { name: c.name, arguments: c.args },
    }));
  }
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIStopOf(ans.stop), logprobs: null }],
    usage: {
      prompt_tokens: ans.inputTokens,
      completion_tokens: ans.outputTokens,
      total_tokens: ans.inputTokens + ans.outputTokens,
    },
  };
}

/** A normalized answer as an OpenAI SSE stream naming `model`. */
function renderOpenAIStream(ans: NormAns, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  if (ans.text !== '') frames.push(sseEvent(null, chunk({ content: ans.text }, null)));
  ans.toolCalls.forEach((call, i) => {
    frames.push(
      sseEvent(
        null,
        chunk(
          {
            tool_calls: [
              {
                index: i,
                id: call.id !== '' ? call.id : `call_${i}`,
                type: 'function',
                function: { name: call.name, arguments: call.args },
              },
            ],
          },
          null,
        ),
      ),
    );
  });
  frames.push(sseEvent(null, chunk({}, openAIStopOf(ans.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: ans.inputTokens,
        completion_tokens: ans.outputTokens,
        total_tokens: ans.inputTokens + ans.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as an Anthropic Messages object naming `model`. */
function renderAnthropicJson(ans: NormAns, model: string): Json {
  const content: Json[] = [];
  if (ans.text !== '' || ans.toolCalls.length === 0) content.push({ type: 'text', text: ans.text });
  for (const call of ans.toolCalls) {
    content.push({
      type: 'tool_use',
      id: call.id !== '' ? call.id : 'toolu_unknown',
      name: call.name,
      input: argsObjectOf(call.args),
    });
  }
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopOf(ans.stop),
    stop_sequence: null,
    usage: { input_tokens: ans.inputTokens, output_tokens: ans.outputTokens },
  };
}

/** A normalized answer as an Anthropic SSE stream naming `model`. */
function renderAnthropicStream(ans: NormAns, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: ans.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let nextIndex = 0;
  frames.push(
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index: nextIndex,
      content_block: { type: 'text', text: '' },
    }),
  );
  const textIndex = nextIndex;
  nextIndex += 1;
  if (ans.text !== '') {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: textIndex,
        delta: { type: 'text_delta', text: ans.text },
      }),
    );
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
  for (const call of ans.toolCalls) {
    const index = nextIndex;
    nextIndex += 1;
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: call.id !== '' ? call.id : `toolu_${index}`,
          name: call.name,
          input: {},
        },
      }),
    );
    if (call.args !== '') {
      frames.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: call.args },
        }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(ans.stop), stop_sequence: null },
      usage: { output_tokens: ans.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

// ---------------------------------------------------------------------------
// Routing and the upstream call
// ---------------------------------------------------------------------------

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const UPSTREAM_TIMEOUT_MS = 8000;
const REQUEST_DEADLINE_MS = 25000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** Providers whose free-tier budget is spent or that are out: never called again. */
const exhausted = new Set<string>();
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that meets the requirements, best rating first. An OpenAI request
 * carrying a response_format can only go to a wire that can carry it (OpenAI or
 * Gemini); the Anthropic wire has no JSON mode, so its providers are not candidates
 * for those requests.
 */
function candidatesFor(requires: Requirement, jsonCarry: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (exhausted.has(probe.provider.id)) continue;
    if (jsonCarry && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingRow(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  const manifest = brief.manifest;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest holds no row for this request: read the requirements off the body.
  const norm = clientApi === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
  let vision = false;
  for (const turn of norm.turns) if (turn.image !== null) vision = true;
  return {
    tools: norm.tools.length > 0,
    vision,
    json: norm.jsonMode,
  };
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  text: string;
  contentType: string;
}

/** One call to a provider in its own wire. */
async function callUpstream(
  candidate: Candidate,
  norm: NormReq,
  modelId: string,
): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = toOpenAI(norm, modelId);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    upstreamBody = toAnthropic(norm, modelId);
  } else {
    url = norm.stream
      ? `${base}/v1beta/models/${encodeURIComponent(modelId)}:streamGenerateContent?alt=sse`
      : `${base}/v1beta/models/${encodeURIComponent(modelId)}:generateContent`;
    upstreamBody = toGemini(norm);
  }
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (norm.stream) headers.accept = 'text/event-stream';
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(upstreamBody),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
  const text = await res.text();
  return { ok: res.ok, status: res.status, text, contentType: res.headers.get('content-type') ?? 'application/json' };
}

/** The failure's message, when the provider answered one as JSON. */
function failureMessage(result: UpstreamResult): string {
  try {
    const value: unknown = JSON.parse(result.text);
    if (isRecord(value)) {
      if (isRecord(value.error) && typeof value.error.message === 'string') return value.error.message;
      if (typeof value.message === 'string') return value.message;
    }
  } catch {
    // Not JSON: no message to read.
  }
  return '';
}

/** Whether a refusal says the provider's budget is spent or it is out: never call it again. */
function isExhausting(result: UpstreamResult): boolean {
  const message = failureMessage(result);
  return /budget is spent/i.test(message) || /\bis out\b/i.test(message);
}

/** A provider call's outcome: success carries the answer to forward or render. */
type Attempt =
  | { kind: 'success'; result: UpstreamResult }
  | { kind: 'exhausted'; result: UpstreamResult }
  | { kind: 'transient'; result: UpstreamResult | null };

/**
 * One attempt at one candidate, with one immediate retry on a transient refusal.
 * A transient refusal (a scheduled fail_call, a 5xx, a network error) does not spend
 * the provider's budget, so retrying the same provider is the right move; a second
 * transient refusal moves on to the next candidate and leaves the provider in play
 * for later requests.
 */
async function attemptCandidate(
  candidate: Candidate,
  norm: NormReq,
  modelId: string,
  deadline: number,
): Promise<Attempt> {
  let result: UpstreamResult | null = null;
  for (let n = 0; n < 2; n++) {
    try {
      result = await callUpstream(candidate, norm, modelId);
    } catch (err) {
      log(`${candidate.probe.provider.id}/${modelId} call failed: ${(err as Error).message}`);
      result = null;
    }
    if (result !== null && result.ok) return { kind: 'success', result };
    if (result !== null && result.status === 400) return { kind: 'transient', result };
    if (result !== null && result.status === 401) return { kind: 'transient', result };
    if (result !== null && isExhausting(result)) return { kind: 'exhausted', result };
    // Transient: retry once, then give up on this candidate for this request.
    if (n === 0 && Date.now() < deadline) continue;
    return { kind: 'transient', result };
  }
  return { kind: 'transient', result };
}

// ---------------------------------------------------------------------------
// The client wires: serving one request
// ---------------------------------------------------------------------------

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The free models across the providers that answered, deduped by id: what GET /v1/models lists. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingRow(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({
        type: 'model',
        id: m.id,
        display_name: m.display_name,
        created_at: FIXED_CREATED_AT,
      })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

/**
 * Serve one client request (either wire) through the routing loop, and answer in the
 * client's own wire. A same-wire provider answer goes back verbatim; a cross-wire one
 * is parsed and re-rendered, so the text, usage and stop family all survive.
 */
async function serve(
  req: IncomingMessage,
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const requires = requirementsFor(body, clientApi);
  const norm = clientApi === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
  const jsonCarry = clientApi === 'openai' && norm.jsonMode;
  const stream = norm.stream;
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : '';
  const deadline = Date.now() + REQUEST_DEADLINE_MS;
  const candidates = candidatesFor(requires, jsonCarry);
  for (const candidate of candidates) {
    if (exhausted.has(candidate.probe.provider.id)) continue;
    if (Date.now() >= deadline) break;
    const attempt = await attemptCandidate(candidate, norm, candidate.model.id, deadline);
    if (attempt.kind === 'exhausted') {
      exhausted.add(candidate.probe.provider.id);
      log(`${candidate.probe.provider.id} is out of budget: not called again`);
      continue;
    }
    if (attempt.kind !== 'success') continue;
    const result = attempt.result;
    const wire = candidate.probe.provider.wire;
    const modelId = requestedModel !== '' ? requestedModel : candidate.model.id;
    // Same wire: the provider's answer is already in the client's shape. Forward it verbatim.
    if (
      (clientApi === 'openai' && wire === 'openai') ||
      (clientApi === 'anthropic' && wire === 'anthropic')
    ) {
      res.writeHead(result.status, { 'content-type': result.contentType, 'content-length': Buffer.byteLength(result.text) });
      res.end(result.text);
      return;
    }
    // Cross-wire: parse the provider's answer and render it in the client's wire.
    const ans = parseProviderAnswer(wire, result.text, stream);
    if (ans === null) {
      log(`${candidate.probe.provider.id}/${candidate.model.id} gave an unparseable answer; trying the next candidate`);
      continue;
    }
    if (clientApi === 'openai') {
      if (stream) sendSse(res, renderOpenAIStream(ans, modelId));
      else sendJson(res, 200, renderOpenAIJson(ans, modelId));
    } else {
      if (stream) sendSse(res, renderAnthropicStream(ans, modelId));
      else sendJson(res, 200, renderAnthropicJson(ans, modelId));
    }
    return;
  }
  return sendError(
    res,
    clientApi,
    503,
    clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model',
    'no free model on the directory meets this request',
  );
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

// ---------------------------------------------------------------------------
// Register, poll, start
// ---------------------------------------------------------------------------

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') {
    log(`registered: the workload holds ${reply.requests} requests`);
  }
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built
    // object and stop before any server starts. A process whose stdout is not flushed
    // loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
