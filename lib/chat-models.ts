export interface ChatModel {
    id: string
    name: string
    icon: string
    free: boolean
}

export const CHAT_MODELS: ChatModel[] = [
    { id: "gemini-2.0-flash", name: "Gemini 2.0 Flash (Recommended)", icon: "✨", free: true },
    { id: "gemini-1.5-flash", name: "Gemini 1.5 Flash (Fast)", icon: "⚡", free: true },
]

export const DEFAULT_MODEL = "gemini-2.0-flash"

export function getModelById(id: string): ChatModel | undefined {
    return CHAT_MODELS.find((m) => m.id === id)
}
