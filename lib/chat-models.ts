export interface ChatModel {
    id: string
    name: string
    icon: string
    free: boolean
}

export const CHAT_MODELS: ChatModel[] = [
    { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", icon: "🚀", free: true },
    { id: "gemini-2.5", name: "Gemini 2.5", icon: "✨", free: true },
    { id: "openrouter/auto", name: "OpenRouter Auto", icon: "🤖", free: true },
]

export const DEFAULT_MODEL = "gemini-2.5-flash"

export function getModelById(id: string): ChatModel | undefined {
    return CHAT_MODELS.find((m) => m.id === id)
}
