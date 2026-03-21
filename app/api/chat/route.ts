import { auth } from "@/auth"
import { CHAT_MODELS, DEFAULT_MODEL } from "@/lib/chat-models"
import { db } from "@/lib/db"
import { type NextRequest, NextResponse } from "next/server"

interface ChatMessage {
  role: "user" | "assistant" | "system"
  content: string
}

interface EnhancePromptRequest {
  prompt: string
  context?: {
    fileName?: string
    language?: string
    codeContent?: string
  }
}

// ─── OpenRouter helper ────────────────────────────────────────────────────

async function callOpenRouter(
  model: string,
  messages: ChatMessage[],
  options: { temperature?: number; max_tokens?: number; timeout?: number } = {}
): Promise<{ content: string; model: string }> {
  const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY
  if (!OPENROUTER_API_KEY) {
    throw new Error("Missing OPENROUTER_API_KEY in environment variables")
  }

  const { temperature = 0.7, max_tokens = 1000, timeout = 30000 } = options

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)

  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      },
      body: JSON.stringify({
        model,
        messages,
        temperature,
        max_tokens,
      }),
      signal: controller.signal,
    })

    clearTimeout(timeoutId)

    if (!response.ok) {
      const errorText = await response.text()
      console.error(`OpenRouter error (${model}):`, errorText)
      throw new Error(`OpenRouter API error ${response.status}: ${errorText}`)
    }

    const data = await response.json()
    const content = data.choices?.[0]?.message?.content?.trim() || ""

    if (!content) {
      throw new Error(`Empty response from ${model}`)
    }

    return { content, model }
  } catch (error) {
    clearTimeout(timeoutId)
    if ((error as Error).name === "AbortError") {
      throw new Error(`Timeout: ${model} took too long to respond`)
    }
    throw error
  }
}

// ─── Generate AI response (single model) ──────────────────────────────────────

async function generateAIResponse(
  messages: ChatMessage[],
  selectedModel: string
): Promise<{ content: string; model: string }> {
  const systemPrompt = `You are an expert AI coding assistant. You help developers with:
- Code explanations and debugging
- Best practices and architecture advice
- Writing clean, efficient code
- Troubleshooting errors
- Code reviews and optimizations

Always provide clear, practical answers. When showing code, use proper formatting with language-specific syntax.
Keep responses concise but comprehensive. Use code blocks with language specification when providing code examples.`

  const fullMessages: ChatMessage[] = [
    { role: "system", content: systemPrompt },
    ...messages,
  ]

  const selectedModelIds =
    selectedModel === "all"
      ? CHAT_MODELS.map((m) => m.id)
      : [selectedModel || DEFAULT_MODEL]

  let lastError: Error | null = null

  for (const modelId of selectedModelIds) {
    try {
      return await callOpenRouter(modelId, fullMessages, {
        temperature: 0.7,
        max_tokens: 1000,
        timeout: 30000,
      })
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err))
      console.warn(`[AI Chat] OpenRouter ${modelId} failed, trying next model: ${lastError.message}`)
    }
  }

  throw new Error(
    `No OpenRouter model succeeded (selected: ${selectedModel}). last error: ${lastError?.message || "unknown"}`
  )
}

// ─── Enhance prompt ───────────────────────────────────────────────────────────

async function enhancePrompt(request: EnhancePromptRequest) {
  const enhancementPrompt = `You are a prompt enhancement assistant. Take the user's basic prompt and enhance it to be more specific, detailed, and effective for a coding AI assistant.

Original prompt: "${request.prompt}"

Context: ${request.context ? JSON.stringify(request.context, null, 2) : "No additional context"}

Enhanced prompt should:
- Be more specific and detailed
- Include relevant technical context
- Ask for specific examples or explanations
- Be clear about expected output format
- Maintain the original intent

Return only the enhanced prompt, nothing else.`

  try {
    const result = await callOpenRouter(
      DEFAULT_MODEL,
      [{ role: "user", content: enhancementPrompt }],
      { temperature: 0.3, max_tokens: 500, timeout: 15000 }
    )
    return result.content || request.prompt
  } catch (error) {
    console.error("Prompt enhancement error:", error)
    return request.prompt
  }
}

// ─── Save message to MongoDB ──────────────────────────────────────────────────

async function saveMessage(
  userId: string,
  role: string,
  content: string,
  playgroundId?: string,
  model?: string
) {
  try {
    await db.chatMessage.create({
      data: {
        userId,
        role,
        content,
        playgroundId: playgroundId || null,
        model: model || null,
      },
    })
  } catch (error) {
    console.error("Failed to save message to DB:", error)
    // Don't throw — saving to DB should not break the chat
  }
}

// ─── Load past messages for AI context ────────────────────────────────────────

async function loadPastMessages(
  userId: string,
  playgroundId?: string,
  limit = 20
): Promise<ChatMessage[]> {
  try {
    const dbMessages = await db.chatMessage.findMany({
      where: {
        userId,
        ...(playgroundId ? { playgroundId } : {}),
      },
      orderBy: { createdAt: "desc" },
      take: limit,
    })

    // Reverse so oldest first, and map to ChatMessage format
    return dbMessages.reverse().map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    }))
  } catch (error) {
    console.error("Failed to load past messages:", error)
    return []
  }
}

// ─── POST handler ─────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()

    // Handle prompt enhancement
    if (body.action === "enhance") {
      const enhancedPrompt = await enhancePrompt(body as EnhancePromptRequest)
      return NextResponse.json({ enhancedPrompt })
    }

    // Handle regular chat
    const { message, history, model: requestedModel, playgroundId } = body

    if (!message || typeof message !== "string") {
      return NextResponse.json({ error: "Message is required and must be a string" }, { status: 400 })
    }

    // Resolve model
    const selectedModel = requestedModel || DEFAULT_MODEL

    // Get authenticated user for persistence
    const session = await auth()
    const userId = session?.user?.id

    // Build conversation context: combine DB history + recent client history
    let contextMessages: ChatMessage[] = []

    if (userId) {
      // Load past messages from MongoDB for AI memory
      const pastMessages = await loadPastMessages(userId, playgroundId, 20)
      contextMessages = [...pastMessages]
    }

    // Also include any recent client-side history not yet in DB
    const validHistory = Array.isArray(history)
      ? (history as ChatMessage[]).filter(
        (msg) =>
          msg &&
          typeof msg === "object" &&
          typeof msg.role === "string" &&
          typeof msg.content === "string" &&
          ["user", "assistant"].includes(msg.role),
      )
      : []

    // Merge: use DB history as base, append any client-only messages
    const recentHistory = validHistory.slice(-5) // Last 5 from client as supplement
    const messages: ChatMessage[] = [
      ...contextMessages,
      ...recentHistory,
      { role: "user", content: message },
    ]

    // Deduplicate consecutive messages with same content
    const deduped = messages.filter(
      (msg, i) => i === 0 || msg.content !== messages[i - 1].content
    )

    const aiResult = await generateAIResponse(deduped, selectedModel)

    if (!aiResult.content) {
      throw new Error("Empty response from AI model")
    }

    // Save both user message and AI response to MongoDB
    if (userId) {
      await saveMessage(userId, "user", message, playgroundId)
      await saveMessage(userId, "assistant", aiResult.content, playgroundId, aiResult.model)
    }

    return NextResponse.json({
      response: aiResult.content,
      model: aiResult.model,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("Error in AI chat route:", error)
    const errorMessage = error instanceof Error ? error.message : "Unknown error occurred"
    return NextResponse.json(
      {
        error: "Failed to generate AI response",
        details: errorMessage,
        timestamp: new Date().toISOString(),
      },
      { status: 500 },
    )
  }
}

// ─── GET handler ──────────────────────────────────────────────────────────────

export async function GET() {
  return NextResponse.json({
    status: "AI Chat API is running",
    timestamp: new Date().toISOString(),
    models: CHAT_MODELS,
    info: "Use POST method to send chat messages or enhance prompts",
  })
}
