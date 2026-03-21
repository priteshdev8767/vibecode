/**
 * Ultra-Fast AI Inline Code Suggestion API
 *
 * Model: OpenRouter Auto
 *
 * Capabilities:
 *  • Full-file semantic context (imports, symbols, types, conventions)
 *  • Fill-in-the-Middle (FIM) prompting — prefix + suffix aware
 *  • Multi-candidate generation + confidence scoring
 *  • LRU cache (128 slots, 30s TTL) — skips model on cache hit
 *  • Request deduplication — concurrent identical requests share one fetch
 *  • Per-language stop sequences — prevents over-generation
 *  • Indent-style auto-detection (tabs vs spaces, 2 vs 4)
 *  • Framework-aware prompting (React, Next.js, Vue, Express, Django, …)
 */

import { type NextRequest, NextResponse } from "next/server";

// ─── Configuration ────────────────────────────────────────────────────────────

const CFG = {
  MAX_FILE_BYTES: 150_000,
  CONTEXT_LINES_BEFORE: 25,      // increased for better semantic understanding
  CONTEXT_LINES_AFTER: 15,       // increased for completion context
  MAX_TOKENS: 75,                // optimal for inline completions
  TEMPERATURE_FAST: 0.01,        // ultra-deterministic for consistency
  TEMPERATURE_REASONING: 0.03,   // more deterministic reasoning
  NUM_CANDIDATES: 1,             // single best suggestion
  AI_TIMEOUT_MS: 1_200,          // 1.2s for faster response
  FALLBACK_TIMEOUT_MS: 800,      // 0.8s fallback
  CACHE_SIZE: 1024,              // 2x cache for more hits
  CACHE_TTL_MS: 180_000,         // 3 minute cache (longer reuse)
} as const

// ─── Model Stack ──────────────────────────────────────────────────────────────

type ModelEntry = { id: string; label: string }

const MODEL_STACK: ModelEntry[] = [
  {
    id: "openrouter/auto",
    label: "OpenRouter Auto (Rate-Limited Optimized)",
  },
]

/** Return the ordered list of models to try for a given suggestionType. */
function modelsForType(type: SuggestionType): ModelEntry[] {
  // Smart model selection based on completion type
  switch (type) {
    case "inline":
      // Fast, lightweight model for quick completions
      return [{
        id: "nvidia/nemotron-3-nano-30b-a3b:free",
        label: "Nvidia Nemotron Nano 9B (Free - Optimized for Speed)",
      }]
    case "block":
      // More capable model for complex completions
      return [{
        id: "openrouter/auto",
        label: "OpenRouter Auto (Full Context)",
      }]
    default:
      // Default fallback
      return MODEL_STACK
  }
}

// ─── Types ────────────────────────────────────────────────────────────────────

type SuggestionType = "inline" | "block" | "docstring" | "test" | "refactor" | "debug"

interface CodeSuggestionRequest {
  fileContent: string
  cursorLine: number
  cursorColumn: number
  suggestionType: SuggestionType
  fileName?: string
  /** Other open files — used for cross-file symbol awareness */
  relatedFiles?: { name: string; content: string }[]
  stream?: boolean
}

interface SemanticContext {
  imports: string[]
  exportedSymbols: string[]
  localSymbols: string[]
  nearbyFunctionSignatures: string[]
  typeDefinitions: string[]
  currentScope: "function" | "class" | "module"
  indentStyle: "spaces" | "tabs"
  indentSize: number
}

interface CodeContext {
  language: string
  framework: string
  database: string
  runtime: string
  prefix: string      // everything up to cursor (FIM)
  suffix: string      // everything after cursor (FIM)
  currentLine: string
  cursorPosition: { line: number; column: number }
  isInFunction: boolean
  isInClass: boolean
  isAfterComment: boolean
  incompletePatterns: string[]
  semantic: SemanticContext
}

interface Candidate {
  text: string
  stopReason: string
  confidence: number
  modelUsed: string
}

interface SuggestionResponse {
  suggestion: string
  candidates: Candidate[]
  cached: boolean
  modelUsed: string
  metadata: {
    language: string
    framework: string
    database: string
    runtime: string
    scope: string
    position: { line: number; column: number }
    tokenBudgetUsed: number
    generatedAt: string
    latencyMs: number
  }
}

// ─── LRU Cache ────────────────────────────────────────────────────────────────

interface CacheEntry { response: SuggestionResponse; expiresAt: number }

class LRUCache {
  private map = new Map<string, CacheEntry>()
  constructor(private maxSize: number) { }

  get(key: string): SuggestionResponse | null {
    const entry = this.map.get(key)
    if (!entry) return null
    if (Date.now() > entry.expiresAt) { this.map.delete(key); return null }
    this.map.delete(key)
    this.map.set(key, entry)
    return entry.response
  }

  set(key: string, value: SuggestionResponse): void {
    if (this.map.size >= this.maxSize) {
      this.map.delete(this.map.keys().next().value!)
    }
    this.map.set(key, { response: value, expiresAt: Date.now() + CFG.CACHE_TTL_MS })
  }
}

const cache = new LRUCache(CFG.CACHE_SIZE)
const inFlight = new Map<string, Promise<SuggestionResponse>>()
const lastRequestTime = new Map<string, number>()

// ─── Route Handler ────────────────────────────────────────────────────────────

export async function POST(request: NextRequest) {
  const t0 = Date.now()

  let body: CodeSuggestionRequest
  try { body = await request.json() }
  catch { return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 }) }

  const err = validateRequest(body)
  if (err) return NextResponse.json({ error: err }, { status: 400 })

  const context = analyzeCodeContext(body)
  const cacheKey = buildCacheKey(context, body.suggestionType)

  // ── Smart caching: position-aware with incremental updates ──
  const hit = cache.get(cacheKey)
  if (hit) return NextResponse.json({ ...hit, cached: true })

  // ── Dedup concurrent identical requests ──
  const flying = inFlight.get(cacheKey)
  if (flying) {
    try { return NextResponse.json({ ...(await flying), cached: true }) }
    catch { /* fall through */ }
  }

  const promise = runGeneration(context, body.suggestionType, t0)
  inFlight.set(cacheKey, promise)

  try {
    const result = await promise
    cache.set(cacheKey, result)

    // ── SSE streaming ──
    if (body.stream) {
      const enc = new TextEncoder()
      const tokens = result.suggestion.split(/(?<=\s)|(?=\s)/)
      const stream = new ReadableStream({
        start(ctrl) {
          let i = 0
          const tick = () => {
            if (i >= tokens.length) {
              ctrl.enqueue(enc.encode("data: [DONE]\n\n"))
              ctrl.close()
              return
            }
            ctrl.enqueue(enc.encode(`data: ${JSON.stringify({ token: tokens[i++] })}\n\n`))
            setTimeout(tick, 0)
          }
          tick()
        },
      })
      return new Response(stream, {
        headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
      })
    }

    return NextResponse.json(result)
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error"
    console.error("[CodeSuggestion] Fatal:", message)
    return NextResponse.json({ error: "Internal server error", message }, { status: 500 })
  } finally {
    inFlight.delete(cacheKey)
  }
}

// ─── Validation ───────────────────────────────────────────────────────────────

function validateRequest(b: Partial<CodeSuggestionRequest>): string | null {
  if (!b.fileContent || typeof b.fileContent !== "string") return "fileContent is required"
  if (new TextEncoder().encode(b.fileContent).length > CFG.MAX_FILE_BYTES)
    return `fileContent exceeds ${CFG.MAX_FILE_BYTES / 1000} KB limit`
  if (!Number.isInteger(b.cursorLine) || b.cursorLine! < 0) return "cursorLine must be a non-negative integer"
  if (!Number.isInteger(b.cursorColumn) || b.cursorColumn! < 0) return "cursorColumn must be a non-negative integer"
  if (!b.suggestionType) return "suggestionType is required"
  return null
}

// ─── Context Analysis ─────────────────────────────────────────────────────────

function analyzeCodeContext(body: CodeSuggestionRequest): CodeContext {
  const { fileContent, cursorLine, cursorColumn, fileName, relatedFiles } = body
  const lines = fileContent.split("\n")
  const currentLine = lines[cursorLine] ?? ""

  const prefixStart = Math.max(0, cursorLine - CFG.CONTEXT_LINES_BEFORE)
  const suffixEnd = Math.min(lines.length, cursorLine + CFG.CONTEXT_LINES_AFTER)

  const prefix = lines.slice(prefixStart, cursorLine).join("\n")
    + "\n" + currentLine.substring(0, cursorColumn)
  const suffix = currentLine.substring(cursorColumn)
    + "\n" + lines.slice(cursorLine + 1, suffixEnd).join("\n")

  const language = detectLanguage(fileContent, fileName)
  const framework = detectFramework(fileContent, language)
  const database = detectDatabase(fileContent)
  const runtime = detectRuntime(fileContent, fileName)

  return {
    language,
    framework,
    database,
    runtime,
    prefix,
    suffix,
    currentLine,
    cursorPosition: { line: cursorLine, column: cursorColumn },
    isInFunction: detectScope(lines, cursorLine, "function"),
    isInClass: detectScope(lines, cursorLine, "class"),
    isAfterComment: detectAfterComment(currentLine, cursorColumn),
    incompletePatterns: detectIncompletePatterns(currentLine, cursorColumn),
    semantic: extractSemanticContext(fileContent, lines, cursorLine, language, relatedFiles),
  }
}

// ─── Semantic Extraction ──────────────────────────────────────────────────────

function extractSemanticContext(
  content: string,
  lines: string[],
  cursorLine: number,
  language: string,
  relatedFiles?: { name: string; content: string }[],
): SemanticContext {
  const isPy = language === "Python"
  const isTS = language === "TypeScript"

  // Imports
  const importRe = isPy ? /^(?:import|from)\s+.+/gm : /^(?:import|require)\s+.+/gm
  const imports = [...content.matchAll(importRe)].map(m => m[0].trim()).slice(0, 24)

  // Exported symbols
  const exportRe = /export\s+(?:default\s+)?(?:(?:async\s+)?function|class|const|let|var|type|interface)\s+(\w+)/g
  const exportedSymbols = [...content.matchAll(exportRe)].map(m => m[1])

  // Local symbols visible above cursor
  const localWindow = lines.slice(Math.max(0, cursorLine - 60), cursorLine).join("\n")
  const localSymbols = [...localWindow.matchAll(/(?:const|let|var|function|def|class)\s+(\w+)/g)]
    .map(m => m[1]).slice(-24)

  // Nearby function signatures (80 lines above)
  const fnWindow = lines.slice(Math.max(0, cursorLine - 80), cursorLine).join("\n")
  const sigRe = isPy
    ? /^\s*(?:async\s+)?def\s+\w+\([^)]*\)(?:\s*->\s*\S+)?/gm
    : /^\s*(?:(?:export|default|async|static|private|public|protected|override)\s+)*(?:function\s+\w+|\w+\s*\([^)]*\)\s*(?::\s*[\w<>[\]| ]+)?)\s*(?:\{|$)/gm
  const nearbyFunctionSignatures = [...fnWindow.matchAll(sigRe)]
    .map(m => m[0].trim().replace(/\s*\{$/, ""))
    .slice(-10)

  // Type / interface definitions
  const typeRe = isTS
    ? /^\s*(?:export\s+)?(?:type|interface)\s+\w+[^{]*\{[^}]*\}/gm
    : /^\s*(?:class|@dataclass|TypedDict)\s+\w+/gm
  const typeDefinitions = [...content.matchAll(typeRe)].map(m => m[0].trim()).slice(0, 10)

  // Cross-file symbols from related open files
  if (relatedFiles?.length) {
    for (const rf of relatedFiles.slice(0, 4)) {
      exportedSymbols.push(...[...rf.content.matchAll(exportRe)].map(m => m[1]))
    }
  }

  // Indent detection
  const sample = lines.slice(Math.max(0, cursorLine - 25), cursorLine).filter(l => /^\s+\S/.test(l))
  const usesTab = sample.some(l => l.startsWith("\t"))
  const indentStyle: "spaces" | "tabs" = usesTab ? "tabs" : "spaces"
  const indentSize = usesTab ? 1 : (() => {
    const counts = sample.map(l => l.match(/^( +)/)?.[1].length ?? 0).filter(n => n > 0)
    if (!counts.length) return 2
    const min = Math.min(...counts)
    return [2, 4].includes(min) ? min : 2
  })()

  const currentScope: SemanticContext["currentScope"] = detectScope(lines, cursorLine, "function")
    ? "function"
    : detectScope(lines, cursorLine, "class")
      ? "class"
      : "module"

  return {
    imports: [...new Set(imports)],
    exportedSymbols: [...new Set(exportedSymbols)],
    localSymbols: [...new Set(localSymbols)],
    nearbyFunctionSignatures,
    typeDefinitions,
    currentScope,
    indentStyle,
    indentSize,
  }
}

// ─── Prompt Builder ───────────────────────────────────────────────────────────

const STOP_SEQUENCES: Record<string, string[]> = {
  TypeScript: ["\n\n\n", "// ===", "export default ", "export function ", "export const ", "export class "],
  JavaScript: ["\n\n\n", "// ===", "module.exports", "export default "],
  Python: ["\n\n\n", "\n# ===", "\nclass ", "\nasync def ", "\ndef "],
  Go: ["\n\n\n", "\nfunc ", "\ntype "],
  Java: ["\n\n\n", "\npublic class ", "\nprivate class "],
  default: ["\n\n\n"],
}

function buildPrompt(context: CodeContext, type: SuggestionType): { system: string; user: string } {
  const { prefix, suffix, language, framework, database, runtime, semantic,
    isInFunction, isInClass, isAfterComment, incompletePatterns } = context

  const stack = [
    language,
    framework !== "None" ? framework : null,
    database !== "None" ? database : null,
    runtime !== "None" ? runtime : null,
  ].filter(Boolean).join(" | ")

  const indent = semantic.indentStyle === "tabs"
    ? "tabs"
    : `${semantic.indentSize}-space indentation`

  const hints = [
    semantic.imports.length && `Imports in use:\n${semantic.imports.slice(0, 10).join("\n")}`,
    semantic.nearbyFunctionSignatures.length && `Nearby functions:\n${semantic.nearbyFunctionSignatures.join("\n")}`,
    semantic.typeDefinitions.length && `Types/interfaces:\n${semantic.typeDefinitions.slice(0, 5).join("\n")}`,
    semantic.exportedSymbols.length && `Exported symbols: ${semantic.exportedSymbols.slice(0, 12).join(", ")}`,
    semantic.localSymbols.length && `Local symbols: ${semantic.localSymbols.slice(-14).join(", ")}`,
  ].filter(Boolean).join("\n\n")

  const typeInstructions: Record<SuggestionType, string> = {
    inline: "Complete the current line or statement at <CURSOR> with the most likely next tokens. Focus on immediate context and common patterns. Keep completions short (1-3 words/tokens) and highly probable.",
    block: "Complete the logical code block starting at <CURSOR>. Consider the full context and provide meaningful completion.",
    docstring: "Generate a complete JSDoc / docstring comment for the function at <CURSOR>.",
    test: "Generate a complete unit test for the function at <CURSOR>. Use existing test framework if visible.",
    refactor: "Rewrite the selected code at <CURSOR> to be cleaner, more idiomatic, and more performant.",
    debug: "Identify and fix the bug at or near <CURSOR>. Change only what is required.",
  }

  const system =
    `You are an expert ${stack} code completion AI like GitHub Copilot.\n` +
    `Task: ${typeInstructions[type]}\n\n` +
    `SMART COMPLETION RULES:\n` +
    `- Analyze the immediate context around <CURSOR> for patterns and intent\n` +
    `- Use ${indent} consistently\n` +
    `- Respect ${semantic.currentScope} scope boundaries\n` +
    `- Complete based on: existing imports, variable names, function signatures, and code patterns\n` +
    `- For inline: predict next 1-5 tokens that would naturally follow\n` +
    `- Prefer completions that match the project's coding style and conventions\n` +
    `- Consider language-specific idioms and best practices\n` +
    `- If completing a function call, suggest logical parameter values\n` +
    `- If completing a variable, use appropriate naming conventions\n` +
    `- Output ONLY raw code - no explanations, no markdown, no backticks\n` +
    `- Keep completions concise but complete logical units`

  const user =
    `Language: ${language} | Framework: ${framework} | Scope: ${semantic.currentScope}\n` +
    `Position: Line ${context.cursorPosition.line}, Column ${context.cursorPosition.column}\n` +
    `In Function: ${isInFunction} | In Class: ${isInClass} | After Comment: ${isAfterComment}\n` +
    (hints ? `\n=== Code Context ===\n${hints}\n` : "") +
    `\n=== Code to Complete ===\n${prefix}<CURSOR>${suffix}\n` +
    `\n=== Completion Instructions ===\n` +
    `Generate the most likely ${type} completion that would naturally follow <CURSOR>.\n` +
    `Focus on immediate context and common ${language} patterns.\n` +
    `Complete the current statement or logical unit.`

  return { system, user }
}

// ─── Generation with Model Waterfall ─────────────────────────────────────────

async function runGeneration(
  context: CodeContext,
  type: SuggestionType,
  t0: number,
): Promise<SuggestionResponse> {
  const API_KEY = process.env.OPENROUTER_API_KEY
  if (!API_KEY) throw new Error("OPENROUTER_API_KEY not configured")

  const { system, user } = buildPrompt(context, type)
  const stops = STOP_SEQUENCES[context.language] ?? STOP_SEQUENCES.default
  const tokenBudgetUsed = Math.ceil((system.length + user.length) / 4)
  const temperature = ["inline", "block"].includes(type)
    ? CFG.TEMPERATURE_FAST
    : CFG.TEMPERATURE_REASONING

  const orderedModels = modelsForType(type)

  let lastError: Error | null = null

  for (const model of orderedModels) {
    try {
      const candidates = await callOpenRouter(API_KEY, model.id, system, user, stops, temperature, model.id === orderedModels.at(-1)?.id ? CFG.FALLBACK_TIMEOUT_MS : CFG.AI_TIMEOUT_MS)

      if (!candidates.length) continue

      // Pre-filter obviously bad candidates before scoring
      const filteredCandidates = preFilterCandidates(candidates, context, type)

      if (!filteredCandidates.length) continue

      const scored = filteredCandidates
        .map(c => ({ ...c, modelUsed: model.id, confidence: scoreSuggestion(c, context, type) }))
        .sort((a, b) => b.confidence - a.confidence)

      return {
        suggestion: scored[0]!.text,
        candidates: scored,
        cached: false,
        modelUsed: model.id,
        metadata: {
          language: context.language,
          framework: context.framework,
          database: context.database,
          runtime: context.runtime,
          scope: context.semantic.currentScope,
          position: context.cursorPosition,
          tokenBudgetUsed,
          generatedAt: new Date().toISOString(),
          latencyMs: Date.now() - t0,
        },
      }
    } catch (err) {
      const errorObj = err instanceof Error ? err : new Error(String(err))

      // Copilot-like fallback: if model is unavailable or quota is exceeded, try next model
      if (model.id.startsWith("openrouter") && /404\s*\:\s*\{\s*"error"\s*\:\s*\{\s*"code"\s*\:\s*404/.test(errorObj.message)) {
        console.warn(`[CodeSuggestion] ${model.id} not found for API version; trying next model: ${errorObj.message}`)
        lastError = errorObj
        continue
      }

      if (model.id.startsWith("openrouter") && /429/.test(errorObj.message)) {
        console.warn(`[CodeSuggestion] ${model.id} quota reached, trying next fallback model: ${errorObj.message}`)
        lastError = errorObj
        continue
      }

      if (model.id.startsWith("openrouter") && /429/.test(errorObj.message)) {
        console.warn(`[CodeSuggestion] ${model.id} quota reached, trying next fallback model: ${errorObj.message}`)
        lastError = errorObj
        continue
      }

      lastError = errorObj
      console.warn(`[CodeSuggestion] Model ${model.id} failed: ${lastError.message} — trying next`)
      continue
    }
  }

  // ── Graceful degradation: return empty suggestion instead of 500 error ──
  console.warn(`[CodeSuggestion] All models exhausted, returning empty suggestion. Last error: ${lastError?.message}`)
  return {
    suggestion: "",
    candidates: [],
    cached: false,
    modelUsed: "none",
    metadata: {
      language: context.language,
      framework: context.framework,
      database: context.database,
      runtime: context.runtime,
      scope: context.semantic.currentScope,
      position: context.cursorPosition,
      tokenBudgetUsed: 0,
      generatedAt: new Date().toISOString(),
      latencyMs: Date.now() - t0,
    },
  }
}

async function callOpenRouter(
  apiKey: string,
  modelId: string,
  system: string,
  user: string,
  stops: string[],
  temperature: number,
  timeoutMs: number,
): Promise<Omit<Candidate, "confidence" | "modelUsed">[]> {
  const ctrl = new AbortController()
  const timeout = setTimeout(() => ctrl.abort(), timeoutMs)

  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      signal: ctrl.signal,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: modelId,
        messages: [
          { role: "system", content: system },
          { role: "user", content: user },
        ],
        max_tokens: CFG.MAX_TOKENS,
        temperature,
        n: CFG.NUM_CANDIDATES,
        stop: stops,
      }),
    })

    if (!response.ok) {
      const body = await response.text()
      throw new Error(`HTTP ${response.status}: ${body}`)
    }

    const data = await response.json()
    return (data.choices ?? [])
      .map((c: { message?: { content?: string }; finish_reason?: string }) => ({
        text: cleanSuggestion(c.message?.content ?? ""),
        stopReason: c.finish_reason ?? "unknown",
      }))
      .filter((c: { text: string }) => c.text.length > 0)
  } finally {
    clearTimeout(timeout)
  }
}

// ─── Candidate Scoring ────────────────────────────────────────────────────────

/** Filter out obviously bad suggestions before detailed scoring. */
function preFilterCandidates(candidates: Omit<Candidate, "confidence" | "modelUsed">[], context: CodeContext, type: SuggestionType): Omit<Candidate, "confidence" | "modelUsed">[] {
  return candidates.filter(candidate => {
    const text = candidate.text.trim()

    // Filter out empty or whitespace-only suggestions
    if (!text) return false

    // Filter out suggestions that start with the same word as the current line ends with
    const lastWord = context.prefix.match(/(\w+)\s*$/)?.[1]
    if (lastWord && text.startsWith(lastWord)) return false

    // Filter out suggestions that are just repeating existing code
    if (context.suffix && text === context.suffix.slice(0, text.length)) return false

    // Filter out suggestions with wrong indentation for the file
    const wrongIndentRe = context.semantic.indentStyle === "tabs" ? /^ {4}/m : /^\t/m
    if (wrongIndentRe.test(text)) return false

    // Filter out extremely short suggestions for block completions
    if (type === 'block' && text.length < 5) return false

    return true
  })
}

function detectProgrammingParadigm(context: CodeContext): 'functional' | 'oop' | 'reactive' | 'imperative' {
  const imports = context.semantic.imports.join(' ')
  const symbols = context.semantic.localSymbols.join(' ')

  // Functional indicators
  const functionalScore = (
    (imports.match(/\bmap\b|\bfilter\b|\breduce\b/) || []).length +
    (symbols.match(/=>\s*{/) || []).length * 2 +
    (context.semantic.currentScope === 'function' ? 1 : 0)
  )

  // OOP indicators
  const oopScore = (
    (imports.match(/\bclass\b|\bextends\b|\bimplements\b/) || []).length * 3 +
    (context.isInClass ? 3 : 0) +
    (symbols.match(/\bthis\./) || []).length
  )

  // Reactive indicators
  const reactiveScore = (
    (imports.match(/\bObservable\b|\bSubject\b|\bBehaviorSubject\b/) || []).length * 3 +
    (imports.match(/\bpipe\b|\bof\b|\bfrom\b/) || []).length * 2
  )

  // Determine paradigm
  const maxScore = Math.max(functionalScore, oopScore, reactiveScore)
  if (maxScore === 0) return 'imperative'
  if (functionalScore === maxScore) return 'functional'
  if (oopScore === maxScore) return 'oop'
  return 'reactive'
}

function scoreSuggestion(
  candidate: Omit<Candidate, "confidence" | "modelUsed">,
  context: CodeContext,
  type: SuggestionType,
): number {
  let score = 100

  // Natural stop = model finished cleanly
  if (candidate.stopReason === "stop") score += 20

  // Length heuristic
  const len = candidate.text.length
  if (len < 3) score -= 60
  else if (len < 10) score -= 15
  else if (len > 400) score -= 25

  // Local symbol usage bonus
  score += context.semantic.localSymbols
    .filter(s => candidate.text.includes(s)).length * 6

  // Penalise if suggestion starts by repeating the last identifier in the prefix
  const lastWord = context.prefix.match(/(\w+)\s*$/)?.[1]
  if (lastWord && candidate.text.trimStart().startsWith(lastWord)) score -= 20

  // Indent style penalty
  const wrongIndentRe = context.semantic.indentStyle === "tabs" ? /^ {4}/m : /^\t/m
  if (wrongIndentRe.test(candidate.text)) score -= 20

  // Penalise hallucinated imports (symbols not found anywhere in file)
  const newImportMatch = candidate.text.match(/import\s+.+\s+from\s+['"]([^'"]+)['"]/g)
  if (newImportMatch) {
    for (const imp of newImportMatch) {
      const pkg = imp.match(/from\s+['"]([^'"]+)['"]/)?.[1] ?? ""
      const alreadyImported = context.semantic.imports.some(i => i.includes(pkg))
      if (!alreadyImported) score -= 30
    }
  }

  // Boost for completions that match current indentation pattern
  const currentIndent = context.prefix.match(/^(\s*)/)?.[1] || ""
  if (candidate.text.startsWith(currentIndent)) {
    score += 10
  }

  // Boost for completions that continue logical code patterns
  if (context.language === 'typescript' || context.language === 'javascript') {
    // Arrow functions and method completions
    if (candidate.text.includes('=>') || candidate.text.includes('function')) {
      score += 8
    }
    // Type annotations
    if (candidate.text.includes(': ') || candidate.text.includes('<')) {
      score += 5
    }
  }

  // Context-aware boosts for different suggestion types
  if (type === 'block') {
    // Prefer multi-line completions for blocks
    if (candidate.text.includes('\n')) {
      score += 15
    }
    // Penalize very short block suggestions
    if (candidate.text.trim().length < 15) {
      score -= 20
    }
  } else if (type === 'inline') {
    // Prefer single-line completions for inline
    if (!candidate.text.includes('\n')) {
      score += 10
    }
  }

  // Programming paradigm detection and scoring
  const paradigm = detectProgrammingParadigm(context)
  if (paradigm === 'functional' && candidate.text.includes('=>')) {
    score += 12
  } else if (paradigm === 'oop' && candidate.text.includes('class ')) {
    score += 12
  } else if (paradigm === 'reactive' && candidate.text.includes('Observable')) {
    score += 12
  }

  return score
}

// ─── Suggestion Cleanup ───────────────────────────────────────────────────────

function cleanSuggestion(raw: string): string {
  let s = raw

  // Strip markdown fences
  if (s.includes("```")) {
    const match = s.match(/```[\w-]*\n?([\s\S]*?)```/)
    s = match ? match[1] : s.replace(/```[\w-]*/g, "")
  }

  // Remove cursor artefacts
  s = s.replace(/\|CURSOR\|/g, "").replace(/<CURSOR>/g, "")

  // Strip leading/trailing blank lines
  s = s.replace(/^\n+/, "").replace(/\n+$/, "")

  // Drop any prose preamble lines (lines with no code-like characters)
  const lines = s.split("\n")
  const firstCodeIdx = lines.findIndex(l => /[\w"'`({[<@*/\\#!$%&|=]/.test(l))
  if (firstCodeIdx > 0) s = lines.slice(firstCodeIdx).join("\n")

  return s.trim()
}

// ─── Cache Key ────────────────────────────────────────────────────────────────

function buildCacheKey(ctx: CodeContext, type: string): string {
  // Smart cache key: position-aware with semantic context
  const positionKey = `${ctx.cursorPosition.line}:${ctx.cursorPosition.column}`
  const semanticKey = `${ctx.semantic.currentScope}:${ctx.isInFunction}:${ctx.isInClass}`
  const contextHash = `${ctx.prefix.slice(-150)}|${ctx.suffix.slice(0, 50)}`
  return `${ctx.language}:${type}:${positionKey}:${semanticKey}:${contextHash}`
}

// ─── Language Detection ───────────────────────────────────────────────────────

const EXT_MAP: Record<string, string> = {
  ts: "TypeScript", tsx: "TypeScript",
  js: "JavaScript", jsx: "JavaScript", mjs: "JavaScript", cjs: "JavaScript",
  py: "Python", pyw: "Python",
  java: "Java",
  kt: "Kotlin", kts: "Kotlin",
  go: "Go",
  rs: "Rust",
  php: "PHP",
  cs: "C#",
  rb: "Ruby",
  swift: "Swift",
  cpp: "C++", cc: "C++", cxx: "C++",
  c: "C", h: "C",
  css: "CSS", scss: "SCSS", less: "LESS",
  sql: "SQL",
  sh: "Shell", bash: "Shell", zsh: "Shell",
  yaml: "YAML", yml: "YAML",
  json: "JSON",
  html: "HTML", htm: "HTML",
  vue: "Vue",
  svelte: "Svelte",
}

function detectLanguage(content: string, fileName?: string): string {
  if (fileName) {
    const ext = fileName.split(".").pop()?.toLowerCase() ?? ""
    if (EXT_MAP[ext]) return EXT_MAP[ext]
  }
  if (/:\s*(string|number|boolean|void)\b/.test(content) || content.includes(": string")) return "TypeScript"
  if (/^func\s+\w+/.test(content) && content.includes("package ")) return "Go"
  if (/^fn\s+\w+/.test(content) && content.includes("let mut")) return "Rust"
  if (/^\s*def\s+\w+/.test(content) && content.includes("self")) return "Python"
  if (content.includes("public class ") || content.includes("System.out")) return "Java"
  if (content.includes("using System") || /^\s*namespace\s+/.test(content)) return "C#"
  return "JavaScript"
}

// ─── Framework Detection ─────────────────────────────────────────────────────

function detectFramework(content: string, language: string): string {
  if (content.includes("next/") || content.includes("getServerSideProps")) return "Next.js"
  if (content.includes("from 'hono'") || content.includes("new Hono()")) return "Hono"
  if (content.includes("express()") || content.includes("from 'express'")) return "Express.js"
  if (content.includes("import React") || content.includes("useState")) return "React"
  if (content.includes("createApp") || content.includes("<template>")) return "Vue"
  if (content.includes("@angular/") || content.includes("@Component(")) return "Angular"
  if (content.includes("<script lang") || content.includes("$:")) return "Svelte"
  if (content.includes("fastify()") || content.includes("from 'fastify'")) return "Fastify"
  if (language === "Python") {
    if (content.includes("from django")) return "Django"
    if (content.includes("FastAPI()") || content.includes("from fastapi")) return "FastAPI"
    if (content.includes("Flask(__name__)") || content.includes("from flask")) return "Flask"
  }
  if (language === "Go" && content.includes("gin.Default()")) return "Gin"
  return "None"
}

// ─── Database Detection ───────────────────────────────────────────────────────

function detectDatabase(content: string): string {
  if (content.includes("mongoose") || content.includes("mongodb")) return "MongoDB"
  if (content.includes("from 'pg'") || content.includes("postgres")) return "PostgreSQL"
  if (content.includes("createConnection") || content.includes("mysql")) return "MySQL"
  if (content.includes("supabase") || content.includes("@supabase")) return "Supabase"
  if (content.includes("prisma") || content.includes("@prisma")) return "Prisma"
  if (content.includes("drizzle")) return "Drizzle"
  if (content.includes("sqlite")) return "SQLite"
  if (content.includes("redis")) return "Redis"
  return "None"
}

// ─── Runtime Detection ────────────────────────────────────────────────────────

function detectRuntime(content: string, fileName?: string): string {
  if (content.includes("Bun.") || content.includes("import.meta.env")) return "Bun"
  if (content.includes("Deno.") || content.includes("Deno.serve")) return "Deno"
  if (content.includes("process.env") || content.includes("require(")) return "Node.js"
  if (fileName?.endsWith(".ts") || fileName?.endsWith(".tsx")) return "Node.js"
  return "None"
}

// ─── Scope Detection (brace-counting + Python indent) ────────────────────────

function detectScope(lines: string[], cursorLine: number, kind: "function" | "class"): boolean {
  let depth = 0

  for (let i = cursorLine; i >= 0; i--) {
    const line = lines[i] ?? ""

    for (let c = line.length - 1; c >= 0; c--) {
      const ch = line[c]
      if (ch === "}") { depth++; continue }
      if (ch === "{") {
        if (depth > 0) { depth--; continue }
        const t = line.trimStart()
        if (kind === "function") {
          return (
            /^(?:async\s+)?function\b/.test(t) ||
            /\b(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:\(|function)/.test(t) ||
            /^\w[\w\s,<>]*\(.*\)\s*(?::\s*[\w<>[\]|& ]+)?\s*\{/.test(t) ||
            /=>\s*\{/.test(line) ||
            /^(?:async\s+)?def\s+\w+/.test(t) ||
            /^func\s+\w+/.test(t)
          )
        } else {
          return /^(?:export\s+)?(?:abstract\s+)?(?:class|interface)\s+\w+/.test(t)
        }
      }
    }

    // Python: indent-based scope
    if (i < cursorLine) {
      const t = line.trimStart()
      const matchesDef = kind === "function"
        ? /^(?:async\s+)?def\s+\w+\(/.test(t)
        : /^class\s+\w+/.test(t)
      if (matchesDef) {
        const di = line.match(/^(\s*)/)?.[1].length ?? 0
        const ci = lines[cursorLine]?.match(/^(\s*)/)?.[1].length ?? 0
        if (ci > di) return true
      }
    }
  }

  return false
}

// ─── Comment Detection ────────────────────────────────────────────────────────

function detectAfterComment(line: string, column: number): boolean {
  const before = line.substring(0, column)
  if (/\/\/|#/.test(before)) return true
  const lo = before.lastIndexOf("/*")
  const lc = before.lastIndexOf("*/")
  return lo !== -1 && lo > lc
}

// ─── Incomplete Pattern Detection ────────────────────────────────────────────

function detectIncompletePatterns(line: string, column: number): string[] {
  const before = line.substring(0, column).trimEnd()
  const patterns: string[] = []

  if (/\b(if|while|for|switch)\s*\($/.test(before)) patterns.push("conditional")
  if (/\b(function|def)\s*$/.test(before)) patterns.push("function-declaration")
  if (/\bclass\s*$/.test(before)) patterns.push("class-declaration")
  if (/\btry\s*\{?\s*$/.test(before)) patterns.push("try-catch")
  if (/\bimport\s+\w/.test(before) && !/from/.test(before)) patterns.push("import-from")
  if (/\{\s*$/.test(before)) patterns.push("object-or-block")
  if (/\[\s*$/.test(before)) patterns.push("array-literal")
  if (/(?:^|[^=!<>])=(?!=)\s*$/.test(before)) patterns.push("assignment")
  if (/\.\s*$/.test(before)) patterns.push("method-chain")
  if (/,\s*$/.test(before)) patterns.push("argument-list")
  if (/\(\s*$/.test(before)) patterns.push("function-call")
  if (/=>\s*$/.test(before)) patterns.push("arrow-body")
  if (/:\s*$/.test(before)) patterns.push("type-or-ternary")
  if (/\?\.?\s*$/.test(before)) patterns.push("optional-chain")
  if (/\|\|\s*$|\&\&\s*$/.test(before)) patterns.push("logical-operator")
  if (/return\s*$/.test(before)) patterns.push("return-value")

  return patterns
}