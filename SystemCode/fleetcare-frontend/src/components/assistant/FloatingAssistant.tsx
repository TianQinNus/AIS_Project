import { CloseOutlined, RobotOutlined, SendOutlined } from '@ant-design/icons'
import { Alert, Button, Card, FloatButton, Input, List, Space } from 'antd'
import type { TextAreaRef } from 'antd/es/input/TextArea'
import type { ReactNode } from 'react'
import { useEffect, useRef, useState } from 'react'
import { sendChatMessage } from '../../utils/chat/dataQueries'

type ChatMessage = {
  id: string
  role: 'assistant' | 'user'
  content: string
}

const initialMessages: ChatMessage[] = [
  {
    id: 'welcome',
    role: 'assistant',
    content: 'Hello! I can help you review fleet risk and maintenance priorities.',
  },
]

function createMessageId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function renderInlineMarkdown(text: string): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*)/g)

  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>
    }

    return <span key={`${part}-${index}`}>{part}</span>
  })
}

function MarkdownMessage({ content }: { content: string }) {
  const lines = content.replace(/\r\n/g, '\n').split('\n')
  const blocks: ReactNode[] = []
  let listItems: { content: string; ordered: boolean }[] = []

  const flushList = () => {
    if (listItems.length === 0) return

    const ListTag = listItems[0].ordered ? 'ol' : 'ul'
    blocks.push(
      <ListTag key={`list-${blocks.length}`} style={markdownListStyle}>
        {listItems.map((item, index) => (
          <li key={`${item.content}-${index}`}>{renderInlineMarkdown(item.content)}</li>
        ))}
      </ListTag>,
    )
    listItems = []
  }

  lines.forEach((line, index) => {
    const unorderedListItem = line.match(/^\s*[-*]\s+(.+)$/)
    const orderedListItem = line.match(/^\s*\d+[.)]\s+(.+)$/)
    const listItem = unorderedListItem ?? orderedListItem

    if (listItem) {
      const ordered = orderedListItem !== null
      if (listItems.length > 0 && listItems[0].ordered !== ordered) {
        flushList()
      }
      listItems.push({ content: listItem[1], ordered })
      return
    }

    flushList()
    if (line.trim() === '') return

    const heading = line.match(/^\s*(#{1,6})\s+(.+)$/)
    if (heading) {
      const level = heading[1].length
      const headingTags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] as const
      const HeadingTag = headingTags[level - 1]
      blocks.push(
        <HeadingTag key={`heading-${index}`} style={markdownHeadingStyle}>
          {renderInlineMarkdown(heading[2])}
        </HeadingTag>,
      )
      return
    }

    blocks.push(
      <p key={`paragraph-${index}`} style={markdownParagraphStyle}>
        {renderInlineMarkdown(line)}
      </p>,
    )
  })
  flushList()

  return <>{blocks}</>
}

function FloatingAssistant() {
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState(initialMessages)
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<TextAreaRef>(null)
  const requestControllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (isChatOpen) {
      inputRef.current?.focus()
    }
  }, [isChatOpen])

  useEffect(() => {
    return () => requestControllerRef.current?.abort()
  }, [])

  const sendMessage = async () => {
    const trimmedMessage = message.trim()

    if (!trimmedMessage || isSending) {
      return
    }

    const controller = new AbortController()
    requestControllerRef.current = controller
    setError(null)
    setIsSending(true)
    setMessages((currentMessages) => [...currentMessages, { id: createMessageId(), role: 'user', content: trimmedMessage }])
    setMessage('')

    try {
      const response = await sendChatMessage(trimmedMessage, controller.signal)
      setMessages((currentMessages) => [
        ...currentMessages,
        { id: createMessageId(), role: 'assistant', content: response.answer },
      ])
    } catch (requestError: unknown) {
      if (requestError instanceof DOMException && requestError.name === 'AbortError') return
      setError(requestError instanceof Error ? requestError.message : 'Unable to get a response from the AI assistant')
    } finally {
      if (requestControllerRef.current === controller) {
        requestControllerRef.current = null
        setIsSending(false)
      }
    }
  }

  return (
    <>
      {isChatOpen && (
        <Card
          title={
            <Space size="small">
              <RobotOutlined />
              <span>AI Assistant</span>
            </Space>
          }
          extra={
            <Button
              type="text"
              icon={<CloseOutlined />}
              aria-label="Close AI Assistant"
              onClick={() => setIsChatOpen(false)}
            />
          }
          styles={{ body: chatBodyStyle, header: chatHeaderStyle }}
          style={chatWindowStyle}
        >
          <div style={messageListStyle}>
            <List
              dataSource={messages}
              split={false}
              renderItem={(chatMessage) => (
                <List.Item
                  style={{
                    justifyContent: chatMessage.role === 'user' ? 'flex-end' : 'flex-start',
                    padding: '6px 0',
                  }}
                >
                  <div
                    style={{
                      ...messageBubbleStyle,
                      ...(chatMessage.role === 'user' ? userBubbleStyle : assistantBubbleStyle),
                    }}
                  >
                    {chatMessage.role === 'assistant' ? (
                      <MarkdownMessage content={chatMessage.content} />
                    ) : (
                      <span style={messageContentStyle}>{chatMessage.content}</span>
                    )}
                  </div>
                </List.Item>
              )}
            />
          </div>
          {error && <Alert type="error" showIcon message="Assistant unavailable" description={error} style={errorStyle} />}
          <Space.Compact style={{ width: '100%' }}>
            <Input.TextArea
              ref={inputRef}
              value={message}
              disabled={isSending}
              autoSize={{ minRows: 1, maxRows: 3 }}
              placeholder="Ask about your fleet..."
              aria-label="Message AI Assistant"
              onChange={(event) => setMessage(event.target.value)}
              onPressEnter={(event) => {
                if (!event.shiftKey) {
                  event.preventDefault()
                  sendMessage()
                }
              }}
            />
            <Button type="primary" loading={isSending} icon={<SendOutlined />} aria-label="Send message" onClick={sendMessage} />
          </Space.Compact>
        </Card>
      )}
      <FloatButton
        type="primary"
        icon={isChatOpen ? <CloseOutlined /> : <RobotOutlined />}
        tooltip={isChatOpen ? 'Close AI Assistant' : 'Open AI Assistant'}
        aria-label={isChatOpen ? 'Close AI Assistant' : 'Open AI Assistant'}
        onClick={() => setIsChatOpen((isOpen) => !isOpen)}
        style={floatButtonStyle}
      />
    </>
  )
}

const chatWindowStyle = {
  position: 'fixed' as const,
  right: 24,
  bottom: 88,
  width: 'min(520px, calc(100vw - 32px))',
  height: 'min(680px, calc(100vh - 120px))',
  zIndex: 1000,
  display: 'flex',
  flexDirection: 'column' as const,
  boxShadow: '0 8px 24px rgba(0, 0, 0, 0.16)',
}
const chatHeaderStyle = { padding: '12px 16px' }
const chatBodyStyle = {
  minHeight: 0,
  flex: 1,
  display: 'flex',
  flexDirection: 'column' as const,
  padding: 16,
}
const messageListStyle = { flex: 1, minHeight: 0, overflowY: 'auto' as const, marginBottom: 12 }
const messageBubbleStyle = { maxWidth: '82%', padding: '8px 12px', borderRadius: 8, lineHeight: 1.5 }
const messageContentStyle = { whiteSpace: 'pre-wrap' as const }
const markdownParagraphStyle = { margin: 0 }
const markdownListStyle = { margin: 0, paddingLeft: 20 }
const markdownHeadingStyle = { margin: '8px 0 4px', fontSize: 15 }
const assistantBubbleStyle = { background: '#f0f5ff', color: '#1d39c4' }
const userBubbleStyle = { background: '#1677ff', color: '#fff' }
const errorStyle = { marginBottom: 12 }
const floatButtonStyle = { right: 24, bottom: 24, zIndex: 1001 }

export default FloatingAssistant