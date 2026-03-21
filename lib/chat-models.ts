export interface ChatModel {
    id: string
    name: string
    icon: string
    free: boolean
}

export const CHAT_MODELS: ChatModel[] = [
    { id: "gemini-2.0", name: "Gemini 2.0", icon: "✨", free: true },
    { id: "gemini-1.5", name: "Gemini 1.5", icon: "⚡", free: true },
]

export const DEFAULT_MODEL = "gemini-2.0"

export function getModelById(id: string): ChatModel | undefined {
    return CHAT_MODELS.find((m) => m.id === id)
}
