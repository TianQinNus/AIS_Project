import {
  ArrowLeftOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  CheckOutlined,
} from '@ant-design/icons'
import { Alert, Button, Card, Col, Flex, List, Row, Space, Spin, Tag, Typography } from 'antd'
import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ApiError, getJson } from '../../utils/http'
import { vehicleEndpoints } from '../../utils/Vehicles/endpoints'
import type { VehicleDetail, VehicleTrendPoint } from '../../utils/Vehicles/vehicleDetails'


const { Text, Title } = Typography

const colors = {
  blue: '#1677ff',
  green: '#389e0d',
  red: '#cf1322',
  muted: '#8c8c8c',
}

const riskClasses = ['Risk6', 'Risk12', 'Risk24', 'Risk48', 'Beyond 48'] as const

function formatRiskPercent(value: string) {
  const number = Number(value)
  if (!Number.isFinite(number)) return value
  return `${(number * 100).toFixed(1)}%`
}

const cardTitleStyle = { fontSize: 12, letterSpacing: 0.4, color: colors.muted }

function vehiclesReturnTarget(state: unknown) {
  if (!state || typeof state !== 'object' || !('from' in state)) return null
  const from = (state as { from: unknown }).from
  if (from === '/vehicles' || (typeof from === 'string' && from.startsWith('/vehicles?'))) return from
  return null
}
const sectionCardStyle = { height: '100%' }

function RiskOverTimeChart({ points }: { points: VehicleTrendPoint[] }) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)
  const width = 560
  const height = 260
  const left = 78
  const right = 16
  const top = 16
  const bottom = 78
  const innerWidth = width - left - right
  const innerHeight = height - top - bottom
  const xAt = (index: number) => {
    if (points.length <= 1) return left + innerWidth / 2
    return left + (index / (points.length - 1)) * innerWidth
  }
  const yAt = (riskClass: string) => {
    const index = riskClasses.indexOf(riskClass as (typeof riskClasses)[number])
    const slot = index < 0 ? riskClasses.length - 1 : index
    return top + (slot / (riskClasses.length - 1)) * innerHeight
  }
  const slot = points.length <= 1 ? innerWidth : innerWidth / Math.max(points.length - 1, 1)
  const labelEvery = Math.max(1, Math.ceil(72 / Math.max(slot, 1)))
  const hovered = hoveredIndex === null ? null : points[hoveredIndex]
  const line = points.flatMap((point, index) =>
    point.riskClass ? [`${xAt(index)},${yAt(point.riskClass)}`] : [],
  )

  return (
    <div style={chartFrameStyle}>
      {hovered && hoveredIndex !== null && (
        <div style={tooltipStyle}>
          <Text strong>{hovered.label}</Text>
          {hovered.riskClass && <Text style={{ color: colors.blue }}>{hovered.riskClass}</Text>}
        </div>
      )}
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Predicted risk class over time" style={chartSvgStyle}>
        {riskClasses.map((riskClass) => (
          <g key={riskClass}>
            <line
              x1={left}
              x2={width - right}
              y1={yAt(riskClass)}
              y2={yAt(riskClass)}
              stroke="#f0f0f0"
            />
            <text x={left - 8} y={yAt(riskClass) + 4} fill={colors.muted} fontSize="11" textAnchor="end">
              {riskClass}
            </text>
          </g>
        ))}
        {line.length > 1 && (
          <polyline
            fill="none"
            stroke={colors.blue}
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2.5"
            points={line.join(' ')}
          />
        )}
        {points.map((point, index) => (
          <g key={`${point.label}-${index}`}>
            {point.riskClass && (
              <circle
                cx={xAt(index)}
                cy={yAt(point.riskClass)}
                r={hoveredIndex === index ? 5 : 3.5}
                fill={colors.blue}
              />
            )}
            <rect
              x={index === 0 ? left : (xAt(index - 1) + xAt(index)) / 2}
              y={top}
              width={Math.max(
                (index === points.length - 1 ? width - right : (xAt(index) + xAt(index + 1)) / 2)
                  - (index === 0 ? left : (xAt(index - 1) + xAt(index)) / 2),
                1,
              )}
              height={innerHeight}
              fill="transparent"
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
            />
            {(index % labelEvery === 0 || index === points.length - 1) && (
              <text
                x={xAt(index)}
                y={top + innerHeight + 16}
                fill={colors.muted}
                fontSize="11"
                textAnchor="end"
                transform={`rotate(-30 ${xAt(index)} ${top + innerHeight + 16})`}
              >
                {point.label}
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  )
}

function SummaryMetric({ label, value }: { label: string; value: string }) {
  return (
    <Card style={sectionCardStyle}>
      <Text style={cardTitleStyle}>{label}</Text>
      <Title level={4} style={{ margin: '8px 0 0' }}>
        {value}
      </Title>
    </Card>
  )
}

function VehicleDetailContent({ vehicleId }: { vehicleId: string }) {
  const navigate = useNavigate()
  const location = useLocation()
  const returnTarget = vehiclesReturnTarget(location.state)
  const backToVehicles = () => {
    if (returnTarget) navigate(-1)
    else navigate('/vehicles')
  }
  const [detail, setDetail] = useState<VehicleDetail | null>(null)
  const [status, setStatus] = useState<'loading' | 'ready' | 'missing' | 'error'>('loading')
  const [errorMessage, setErrorMessage] = useState('')

  useEffect(() => {
    let cancelled = false
    setStatus('loading')
    setDetail(null)
    setErrorMessage('')

    getJson<VehicleDetail>(vehicleEndpoints.detail(vehicleId))
      .then((data) => {
        if (!cancelled) {
          setDetail(data)
          setStatus('ready')
        }
      })
      .catch((error: unknown) => {
        if (cancelled) return
        if (error instanceof ApiError && error.status === 404) {
          setStatus('missing')
          return
        }
        setErrorMessage(error instanceof Error ? error.message : 'Unable to load this vehicle')
        setStatus('error')
      })

    return () => {
      cancelled = true
    }
  }, [vehicleId])

  if (status === 'loading') {
    return (
      <Card>
        <Flex justify="center" style={{ padding: 32 }}>
          <Spin />
        </Flex>
      </Card>
    )
  }

  if (status === 'error') {
    return (
      <Card>
        <Space direction="vertical" size={12}>
          <Button type="link" icon={<ArrowLeftOutlined />} style={backButtonStyle} onClick={backToVehicles}>
            Vehicles
          </Button>
          <Alert type="error" showIcon title="Unable to load this vehicle" description={errorMessage} />
        </Space>
      </Card>
    )
  }

  if (!detail) {
    return (
      <Card>
        <Space direction="vertical" size={12}>
          <Button type="link" icon={<ArrowLeftOutlined />} style={backButtonStyle} onClick={backToVehicles}>
            Vehicles
          </Button>
          <Title level={4} style={{ margin: 0 }}>Vehicle not found</Title>
          <Text type="secondary">This vehicle is not in the current fleet view.</Text>
        </Space>
      </Card>
    )
  }

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Button type="link" icon={<ArrowLeftOutlined />} style={backButtonStyle} onClick={backToVehicles}>
        Vehicles
      </Button>

      <Flex justify="space-between" align="center" gap={12} wrap style={{ width: '100%' }}>
        <Title level={3} style={{ margin: 0 }}>{detail.vehicleId}</Title>
        <Tag color={detail.tagColor} style={badgeStyle}>{detail.badge}</Tag>
      </Flex>

      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} lg={8}>
          <SummaryMetric label="Predicted State" value={detail.predictedState} />
        </Col>
        <Col xs={24} sm={12} lg={8}>
          <SummaryMetric label="Action" value={detail.action} />
        </Col>
        <Col xs={24} sm={12} lg={8}>
          <SummaryMetric label="History" value={detail.history} />
        </Col>
      </Row>

      <Card title="CURRENT RISK">
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(5, minmax(0, 1fr))',
            gap: 16,
            width: '100%',
            minWidth: 600,
          }}
        >
          {detail.risks.map((risk) => (
            <div key={risk.label} style={{ textAlign: 'left' }}>
              <Text style={cardTitleStyle}>
                {risk.label}
              </Text>
              <Title level={3} style={{ margin: '8px 0 0' }}>
                {formatRiskPercent(risk.value)}
              </Title>
            </div>
          ))}
        </div>
      </Card>
      <Row gutter={[16, 16]} align="stretch">
        <Col xs={24} lg={12}>
          <Card title="MODEL PREDICTIONS OVER TIME" style={sectionCardStyle}>
            <RiskOverTimeChart points={detail.trend} />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="KEY FACTORS BEHIND THE PREDICTION" style={sectionCardStyle}>
            <Space direction="vertical" size={16} style={{ width: '100%' }}>
              {detail.factors.map((factor) => (
                <div key={factor.name}>
                <Flex justify="space-between" align="center" style={{ marginBottom: 6 }}>
                  <Text>{factor.name}</Text>

                  <Space size={8}>
                    <Text type="secondary">
                      {factor.shapValue != null
                        ? Math.abs(factor.shapValue).toFixed(4)
                        : '—'}
                    </Text>

                    {factor.shapValue != null && (
                      factor.shapValue > 0 ? (
                        <ArrowUpOutlined style={{ color: colors.red }} />
                      ) : factor.shapValue < 0 ? (
                        <ArrowDownOutlined style={{ color: colors.green }} />
                      ) : null
                    )}
                  </Space>
                </Flex>

                <div style={factorTrackStyle} aria-hidden>
                  <div
                    style={{
                      ...factorBarStyle,
                      width: `${Math.round(factor.impact * 100)}%`,
                    }}
                  />
                </div>
                </div>
              ))}
              <Alert
                type="warning"
                showIcon
                title="Features are anonymised; this does not identify a specific mechanical fault."
              />
            </Space>
          </Card>
        </Col>
      </Row>

      <Card title="WHY THIS ACTION?">
        <Space direction="vertical" size={8} style={{ width: '100%' }}>
          <Flex gap={12} align="center" wrap>
            <Tag color={detail.tagColor} style={{ margin: 0 }}>{detail.action}</Tag>
            <Text>Rule {detail.rule}</Text>
          </Flex>
          <List
            split={false}
            dataSource={detail.reasons}
            renderItem={(reason) => (
              <List.Item style={{ padding: '6px 0', justifyContent: 'flex-start' }}>
                <Space>
                  <CheckOutlined style={{ color: colors.green }} />
                  <Text>{reason}</Text>
                </Space>
              </List.Item>
            )}
          />
        </Space>
      </Card>
    </Space>
  )
}

const backButtonStyle = { padding: 0, height: 'auto' }
const badgeStyle = { margin: 0, maxWidth: '100%', padding: '2px 10px', letterSpacing: 0.6, whiteSpace: 'normal' as const }
const chartFrameStyle = { position: 'relative' as const, width: '100%' }
const chartSvgStyle = { display: 'block', width: '100%', height: 'auto' }
const tooltipStyle = {
  position: 'absolute' as const,
  top: 8,
  right: 8,
  zIndex: 1,
  display: 'flex',
  flexDirection: 'column' as const,
  gap: 2,
  padding: '6px 10px',
  border: '1px solid #f0f0f0',
  borderRadius: 6,
  background: '#fff',
}
const factorTrackStyle = {
  flex: 1,
  height: 8,
  borderRadius: 4,
  background: '#f0f0f0',
  overflow: 'hidden' as const,
}
const factorBarStyle = {
  height: '100%',
  borderRadius: 4,
  background: '#cf1322',
}

export default VehicleDetailContent
