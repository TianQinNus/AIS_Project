import { ArrowDownOutlined, ArrowRightOutlined, ArrowUpOutlined } from '@ant-design/icons'
import { Alert, Card, Col, Divider, List, Row, Skeleton, Space, Tag, Typography } from 'antd'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchDashboard } from '../../utils/dashboard/dataQueries'
import type { DashboardResponse } from '../../utils/dashboard/dataQueries'

const { Text, Title } = Typography

const colors = {
  blue: '#1677ff',
  green: '#389e0d',
  orange: '#d46b08',
  red: '#cf1322',
  yellow: '#d4b106',
  purple: '#722ed1',
  muted: '#8c8c8c',
}

const cardTitleStyle = { fontSize: 12, letterSpacing: 0.5, color: colors.muted }
const sectionCardStyle = { height: '100%' }

const riskActionColors: Record<string, string> = {
  CONTINUE: '#7cc7e8',
  MONITOR: '#36a3f7',
  ENHANCED_MONITOR: '#f2c94c',
  INSPECT: '#f2994a',
  PRIORITY_INSPECT: '#eb5757',
  MANUAL_REVIEW: '#9b8afb',
  COLLECT_MORE_DATA: '#a0aec0',
}

const riskSnapshotActions = [
  'PRIORITY_INSPECT',
  'INSPECT',
  'ENHANCED_MONITOR',
  'MONITOR',
  'CONTINUE',
  'MANUAL_REVIEW',
] as const

type RiskSnapshotChartItem = {
  label: string
  count: number
  percentage: number
  chartPercentage: number
  color: string
}

function DashboardContent() {
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()

    fetchDashboard(controller.signal)
      .then(setDashboard)
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === 'AbortError') return
        setError(requestError instanceof Error ? requestError.message : 'Unable to load dashboard data')
      })

    return () => controller.abort()
  }, [])

  if (error) return <Alert type="error" showIcon message="Dashboard unavailable" description={error} />
  if (!dashboard) return <Skeleton active />

  const dashboardKpis = [
    { kind: 'standard', title: 'FLEET OVERVIEW', value: dashboard.fleet.total.toLocaleString(), details: [{ label: `${dashboard.fleet.analyzed.toLocaleString()} Active`, type: 'success' as const }] },
    { kind: 'standard', title: 'AT RISK', value: dashboard.risk.at_risk.toLocaleString(), details: [{ label: `${dashboard.risk.high} High`, type: 'warning' as const }, { label: `${dashboard.risk.critical} Critical`, type: 'danger' as const }] },
    { kind: 'capacity' },
    {
      kind: 'standard',
      title: 'AVG. RISK SCORE',
      value: `${dashboard.risk.average_score.toFixed(2)} /100`,
      details: [],
      trend: dashboard.risk.average_score_change === undefined
        ? null
        : dashboard.risk.average_score_trend === 'UP' || dashboard.risk.average_score_trend === 'DOWN'
          ? {
            direction: dashboard.risk.average_score_trend,
            change: Math.abs(dashboard.risk.average_score_change),
            }
          : null,
    },
  ] as const
  const riskSnapshotByAction = new Map(dashboard.risk_snapshot.map((item) => [item.action, item]))
  const riskSnapshotItems = riskSnapshotActions.map((action) => {
    const item = riskSnapshotByAction.get(action)

    return {
      label: action.replaceAll('_', ' '),
      count: item?.count ?? 0,
      percentage: item?.percentage ?? 0,
      color: riskActionColors[action],
    }
  })
  const riskSnapshotTotal = riskSnapshotItems.reduce((total, item) => total + item.count, 0)
  const riskSnapshotData = riskSnapshotItems.map((item) => ({
    ...item,
    chartPercentage: riskSnapshotTotal > 0 ? (item.count / riskSnapshotTotal) * 100 : 0,
  }))
  const riskClasses = riskSnapshotData.map((item) => ({
    ...item,
    count: item.count.toLocaleString(),
    percentage: `${item.percentage}%`,
  }))
  const riskAlerts = dashboard.top_risk_alerts.map((alert) => ({
    vehicle: `Vehicle ${alert.vehicle_id}`, description: alert.action.replaceAll('_', ' '), className: alert.state, tagColor: alert.priority === 'CRITICAL' ? 'red' : 'orange', risk: `${alert.risk_score.toFixed(2)}%`,
  }))
  const maintenancePlan = [
    { title: 'Immediate', value: dashboard.maintenance_plan.immediate.toLocaleString(), color: colors.red },
    { title: 'Short term', value: dashboard.maintenance_plan.short_term.toLocaleString(), color: colors.orange },
    { title: 'Monitor', value: dashboard.maintenance_plan.monitor.toLocaleString(), color: colors.blue },
    { title: 'Manual review', value: dashboard.maintenance_plan.manual_review.toLocaleString(), color: colors.purple },
  ]

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Row gutter={[16, 16]}>
        {dashboardKpis.map((kpi) => kpi.kind === 'capacity' ? (
          <Col key="maintenance-capacity" xs={24} sm={12} lg={6}>
            <MaintenanceCapacityCard
              capacity={dashboard.maintenance_capacity}
            />
          </Col>
        ) : (
          <Col key={kpi.title} xs={24} sm={12} lg={6}>
            <Card>
              <Text style={cardTitleStyle}>{kpi.title}</Text>
              <Title level={2} style={{ margin: '8px 0 16px', fontSize: 30 }}>
                {kpi.value}
              </Title>
              {'trend' in kpi && kpi.trend && (
                <Text
                  type={kpi.trend.direction === 'DOWN' ? 'success' : kpi.trend.direction === 'UP' ? 'danger' : 'secondary'}
                  style={trendStyle}
                >
                  {kpi.trend.direction === 'DOWN' ? <ArrowDownOutlined /> : <ArrowUpOutlined />}
                  {kpi.trend.change.toFixed(2)}
                </Text>
              )}
              <Space size={12} wrap>
                {kpi.details.map((detail) => (
                  <Text key={detail.label} type={detail.type} style={{ fontSize: 13 }}>
                    {detail.label}
                  </Text>
                ))}
              </Space>
            </Card>
          </Col>
        ))}
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card
            title="RISK SNAPSHOT"
            extra={<Link to="/vehicles">Full analytics <ArrowRightOutlined /></Link>}
            style={sectionCardStyle}
          >
            <Row align="middle" gutter={[32, 24]}>
              <Col xs={24} sm={10}>
                <RiskSnapshotChart data={riskSnapshotData} />
              </Col>
              <Col xs={24} sm={14}>
                <List
                  dataSource={riskClasses}
                  split={false}
                  renderItem={(riskClass) => (
                    <List.Item style={{ padding: '8px 0' }}>
                      <Space>
                        <span style={{ ...legendDotStyle, background: riskClass.color }} />
                        <Text>{riskClass.label}</Text>
                      </Space>
                      <span style={riskSnapshotValueStyle}>
                        <Text strong style={riskSnapshotCountStyle}>{riskClass.count}</Text>
                        <Text type="secondary" style={riskSnapshotPercentageStyle}>{riskClass.percentage}</Text>
                      </span>
                    </List.Item>
                  )}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card
            title="TOP RISK ALERTS"
            style={sectionCardStyle}
          >
            <List
              dataSource={riskAlerts}
              renderItem={(alert) => (
                <List.Item style={{ padding: '1px 0' }}>
                  <List.Item.Meta
                    title={alert.vehicle}
                    description={<span style={{ fontSize: 12 }}>{alert.description}</span>}
                  />
                  <Space direction="vertical" align="end" size={2}>
                    <Tag color={alert.tagColor}>{alert.className}</Tag>
                    <Text strong>{alert.risk}</Text>
                  </Space>
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      <Card title="MAINTENANCE PLAN SUMMARY">
        <Row align="middle">
          {maintenancePlan.map((item, index) => (
            <Col key={item.title} xs={24} sm={12} lg={6}>
              <div style={summaryBlockStyle}>
                {index > 0 && <Divider type="vertical" style={dividerStyle} />}
                <Text type="secondary">{item.title}</Text>
                <Title level={2} style={{ margin: '8px 0 0' }}>
                  {item.value}
                </Title>
                <Text style={{ color: item.color ?? colors.muted }}>vehicles</Text>
              </div>
            </Col>
          ))}
        </Row>
      </Card>
    </Space>
  )
}

type MaintenanceCapacity = {
  used: number
  total: number
  available: number
  utilisation: number
}

function MaintenanceCapacityCard({
  capacity,
}: {
  capacity: MaintenanceCapacity
}) {
  const percentage = Math.min(
    Math.max(capacity.utilisation, 0),
    100
  )

  return (
    <Card
      style={maintenanceCapacityCardStyle}
      styles={{ body: maintenanceCapacityBodyStyle }}
    >
      <Text style={maintenanceCapacityTitleStyle}>
        MAINTENANCE CAPACITY
      </Text>

      <div style={maintenanceCapacityValueStyle}>
        <Text strong style={maintenanceCapacityUsedStyle}>
          {capacity.used}
        </Text>

        <Text style={maintenanceCapacityLimitStyle}>
          / {capacity.total} maintenance slots
        </Text>
      </div>

      <div
        aria-label={`${capacity.utilisation}% capacity utilised`}
        role="progressbar"
        aria-valuemax={capacity.total}
        aria-valuemin={0}
        aria-valuenow={Math.min(capacity.used, capacity.total)}
        style={maintenanceCapacityProgressTrackStyle}
      >
        <div
          style={{
            ...maintenanceCapacityProgressStyle,
            width: `${percentage}%`,
          }}
        />
      </div>

      <div style={maintenanceCapacityFooterStyle}>
        <Text style={maintenanceCapacityUtilisedStyle}>
          {capacity.utilisation.toFixed(1)}% capacity utilised
        </Text>

        <Text style={maintenanceCapacityAvailableStyle}>
          {capacity.available} slots available
        </Text>
      </div>
    </Card>
  )
}

function RiskSnapshotChart({ data }: { data: RiskSnapshotChartItem[] }) {
  const [hoveredItem, setHoveredItem] = useState<RiskSnapshotChartItem | null>(null)
  let startAngle = -90

  const segments = data.map((item) => {
    const endAngle = startAngle + item.chartPercentage * 3.6
    const startRadians = (startAngle * Math.PI) / 180
    const endRadians = (endAngle * Math.PI) / 180
    const startX = 100 + 90 * Math.cos(startRadians)
    const startY = 100 + 90 * Math.sin(startRadians)
    const endX = 100 + 90 * Math.cos(endRadians)
    const endY = 100 + 90 * Math.sin(endRadians)
    const largeArcFlag = endAngle - startAngle > 180 ? 1 : 0
    const path = `M 100 100 L ${startX} ${startY} A 90 90 0 ${largeArcFlag} 1 ${endX} ${endY} Z`

    startAngle = endAngle
    return { item, path }
  })

  return (
    <div style={chartContainerStyle}>
      <svg viewBox="0 0 200 200" role="img" aria-label="Risk distribution chart" style={chartSvgStyle}>
        {segments.map(({ item, path }) => (
          <path
            key={item.label}
            d={path}
            fill={item.color}
            stroke="#fff"
            strokeWidth="1"
            onMouseEnter={() => setHoveredItem(item)}
            onMouseLeave={() => setHoveredItem(null)}
          />
        ))}
      </svg>
      {hoveredItem && (
        <div style={chartTooltipStyle}>
          <strong>{hoveredItem.label}</strong>
          <br />
          {hoveredItem.count.toLocaleString()} vehicles ({hoveredItem.percentage}%)
        </div>
      )}
    </div>
  )
}

const chartContainerStyle = { position: 'relative' as const, width: 168, maxWidth: '100%', margin: '0 auto' }
const chartSvgStyle = { display: 'block', width: '100%', height: 'auto' }
const chartTooltipStyle = {
  position: 'absolute' as const,
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  padding: '6px 8px',
  borderRadius: 4,
  background: 'rgba(0, 0, 0, 0.75)',
  color: '#fff',
  fontSize: 12,
  lineHeight: 1.4,
  textAlign: 'center' as const,
  whiteSpace: 'nowrap' as const,
  pointerEvents: 'none' as const,
}

const legendDotStyle = { width: 8, height: 8, borderRadius: '50%' }
const riskSnapshotValueStyle = { display: 'grid', gridTemplateColumns: 'minmax(56px, auto) 52px', columnGap: 8, alignItems: 'baseline' }
const riskSnapshotCountStyle = { textAlign: 'right' as const }
const riskSnapshotPercentageStyle = { textAlign: 'right' as const }
const summaryBlockStyle = { position: 'relative' as const, minHeight: 88, padding: '4px 20px' }
const dividerStyle = { position: 'absolute' as const, left: 0, top: 0, height: '100%' }
const trendStyle = { marginBottom: 8, fontSize: 13 }
const maintenanceCapacityCardStyle = { height: '100%', borderRadius: 18 }
const maintenanceCapacityBodyStyle = { padding: '18px 16px 16px' }
const maintenanceCapacityTitleStyle = { fontSize: 11, letterSpacing: 0.5, color: '#595959' }
const maintenanceCapacityValueStyle = { display: 'flex', alignItems: 'baseline', gap: 4, margin: '12px 0 12px', minWidth: 0 }
const maintenanceCapacityUsedStyle = { color: '#020000', fontSize: 32, lineHeight: 1 }
const maintenanceCapacityLimitStyle = { color: '#595959', fontSize: 13, minWidth: 0 }
const maintenanceCapacityProgressTrackStyle = { height: 8, overflow: 'hidden' as const, borderRadius: 999, background: '#ededed' }
const maintenanceCapacityProgressStyle = { width: '75%', height: '100%', borderRadius: 999, background: '#68baf0' }
const maintenanceCapacityFooterStyle = { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 6, marginTop: 12 }
const maintenanceCapacityUtilisedStyle = { color: '#595959', fontSize: 11, minWidth: 0 }
const maintenanceCapacityAvailableStyle = { color: '#237a45', fontSize: 11, minWidth: 0, textAlign: 'right' as const }

export default DashboardContent