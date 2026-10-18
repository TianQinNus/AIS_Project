import { API_BASE_URL } from '../../config/api'
import { chatEndpoints } from './endpoints'

export type ChatResponse = {
  answer: string
}

export async function sendChatMessage(message: string, signal?: AbortSignal): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}${chatEndpoints.chat}`, {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
    signal,
  })

  if (!response.ok) {
    throw new Error(`Chat request failed (${response.status})`)
  }

  const data: unknown = await response.json()
  if (
    typeof data !== 'object' ||
    data === null ||
    !('answer' in data) ||
    typeof data.answer !== 'string'
  ) {
    throw new Error('Chat response did not contain a valid answer')
  }

  return { answer: data.answer }
}
