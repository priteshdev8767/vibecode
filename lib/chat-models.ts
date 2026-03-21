export interface ChatModel {
    id: string
    name: string
    icon: string
    free: boolean
}

export const CHAT_MODELS: ChatModel[] = [
    { id: "openrouter/auto", name: "OpenRouter Auto", icon: "🤖", free: true },
    { id: "nvidia/nemotron-3-nano-30b-a3b:free", name: "NVIDIA Nemotron 3 Nano", icon: "⚡", free: true },
    { id: "google/gemini-2.5-flash-lite", name: "Gemini 2.5 Flash Lite", icon: "✨", free: true },
    { id: "deepseek/deepseek-chat-v3-0324", name: "DeepSeek Chat v3", icon: "🔷", free: true },
    { id: "deepseek/deepseek-r1", name: "DeepSeek R1", icon: "🔶", free: true },
    { id: "google/gemini-2.0-flash-lite-001", name: "Gemini 2.0 Flash Lite", icon: "💫", free: true },
    { id: "arcee-ai/trinity-large-preview:free", name: "Trinity Large Preview", icon: "🎯", free: true },
]

export const DEFAULT_MODEL = "openrouter/auto"

export function getModelById(id: string): ChatModel | undefined {
    return CHAT_MODELS.find((m) => m.id === id)
}
