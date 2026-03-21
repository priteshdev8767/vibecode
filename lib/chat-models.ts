export interface ChatModel {
    id: string
    name: string
    icon: string
    free: boolean
}

export const CHAT_MODELS: ChatModel[] = [
    { id: "openrouter/auto", name: "OpenRouter Auto", icon: "🤖", free: true },
    { id: "openrouter/koala-mini", name: "Koala Mini", icon: "🐨", free: true },
    { id: "openrouter/gpt-4o-mini", name: "GPT-4o Mini", icon: "✨", free: true },
    { id: "openrouter/llama-2-7b", name: "LLaMA 2 7B", icon: "🦙", free: true },
]

export const DEFAULT_MODEL = "openrouter/auto"

export function getModelById(id: string): ChatModel | undefined {
    return CHAT_MODELS.find((m) => m.id === id)
}
