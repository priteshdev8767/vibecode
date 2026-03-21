export interface ChatModel {
    id: string
    name: string
    icon: string
    free: boolean
}

export const CHAT_MODELS: ChatModel[] = [
    { id: "openrouter/auto", name: "OpenRouter Auto", icon: "🤖", free: true },
]

export const DEFAULT_MODEL = "openrouter/auto"

export function getModelById(id: string): ChatModel | undefined {
    return CHAT_MODELS.find((m) => m.id === id)
}
