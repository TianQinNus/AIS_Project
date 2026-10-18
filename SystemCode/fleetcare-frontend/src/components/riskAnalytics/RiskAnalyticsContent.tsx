import { Card, Col, List, Progress, Row, Select, Space, Table, Tag, Typography } from 'antd'
import { useState } from 'react'
import {
  componentRiskData,
  fleetSegments,
  riskDistribution,
  featureSignals,
  riskTrendData,
} from '../../utils/RiskAnalytics/dataQueries'

const { Text, Title } = Typography

const componentColumns = [
  { title: 'Component', dataIndex: 'component', key: 'component' },
  {
    title: 'Avg Risk',
    dataIndex: 'avgRisk',
    key: 'avgRisk',
    render: (value: string) => <Text strong>{value}</Text>,
  },
  {
    title: 'Inspect / Priority Inspect',
    dataIndex: 'riskCount',
    key: 'riskCount',
    render: (value: number, record: (typeof componentRiskData)[number]) => (
      <Tag color={record.tagColor}>{value}</Tag>
    ),
  },
]

function RiskDistributionCard() {
  return (
    <Card title="RISK DISTRIBUTION (All Vehicles)" style={fullHeightStyle}>
      <Row align="middle" gutter={[32, 24]}>
        <Col xs={24} sm={10}>
          <div style={donutStyle} aria-label="Risk distribution chart">
            <div style={donutCenterStyle}>
              <Title level={2} style={{ margin: 0 }}>4,256</Title>
              <Text type="secondary">vehicles</Text>
            </div>
          </div>
        </Col>
        <Col xs={24} sm={14}>
          <List
            dataSource={riskDistribution}
            split={false}
            renderItem={(item) => (
              <List.Item style={{ padding: '7px 0' }}>
                <Space>
                  <span style={{ ...legendDotStyle, background: item.color }} />
                  <Text>{item.label}</Text>
                </Space>
                <Text strong>
                  {item.count} <Text type="secondary">{item.percentage}</Text>
                </Text>
              </List.Item>
            )}
          />
        </Col>
      </Row>
    </Card>
  )
}

function RiskTrendCard() {
  const [hoveredPoint, setHoveredPoint] = useState<(typeof riskTrendData)[number] | null>(null)
  const chartPoints = riskTrendData.map((point, index) => ({
    ...point,
    x: 32 + index * 78,
    y: 192 - point.value * 1.7,
  }))

  return (
    <Card
      title="RISK TREND"
      extra={<Select defaultValue="7 days" options={[{ value: '7 days', label: 'Last 7 days' }]} />}
      style={fullHeightStyle}
    >
      <div style={trendChartStyle} aria-label="Risk trend line chart">
        {hoveredPoint && (
          <div style={trendTooltipStyle}>
            <Text strong>{hoveredPoint.date}</Text>
            <Text type="secondary">Risk: {hoveredPoint.value}</Text>
          </div>
        )}
        <svg viewBox="0 0 520 220" role="img" aria-label="Seven day risk trend">
          <g stroke="#f0f0f0" strokeWidth="1">
            <line x1="32" y1="24" x2="500" y2="24" />
            <line x1="32" y1="80" x2="500" y2="80" />
            <line x1="32" y1="136" x2="500" y2="136" />
            <line x1="32" y1="192" x2="500" y2="192" />
          </g>
          <polyline
            fill="none"
            stroke="#1677ff"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="3"
            points={chartPoints.map((point) => `${point.x},${point.y}`).join(' ')}
          />
          <g fill="#1677ff">
            {chartPoints.map((point) => (
              <circle
                key={point.date}
                cx={point.x}
                cy={point.y}
                r="5"
                onMouseEnter={() => setHoveredPoint(point)}
                onMouseLeave={() => setHoveredPoint(null)}
                style={{ cursor: 'pointer' }}
              />
            ))}
          </g>
          <g fill="#8c8c8c" fontSize="11" textAnchor="middle">
            {chartPoints.map((point) => (
              <text key={point.date} x={point.x} y="214">{point.date}</text>
            ))}
          </g>
        </svg>
      </div>
    </Card>
  )
}

function RiskAnalyticsContent() {
  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <RiskDistributionCard />
        </Col>
        <Col xs={24} lg={8}>
          <RiskTrendCard />
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={8}>
          <Card title="RISK BY COMPONENT" style={fullHeightStyle}>
            <Table
              columns={componentColumns}
              dataSource={componentRiskData}
              pagination={false}
              size="small"
              scroll={{ x: 420 }}
            />
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="FLEET SEGMENT" style={fullHeightStyle}>
            <Space direction="vertical" size={20} style={{ width: '100%' }}>
              {fleetSegments.map((segment) => (
                <div key={segment.label}>
                  <Space align="baseline" style={{ width: '100%', justifyContent: 'space-between' }}>
                    <Text>{segment.label}</Text>
                    <Text strong>{segment.count}</Text>
                  </Space>
                  <Progress percent={segment.percent} showInfo={false} strokeColor={segment.color} />
                </div>
              ))}
            </Space>
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="TOP FEATURE SIGNALS" style={fullHeightStyle}>
            <List
              dataSource={featureSignals}
              split={false}
              renderItem={(signal) => (
                <List.Item>
                  <Text>{signal.label}</Text>
                  <Text strong style={{ color: signal.color }}>{signal.status}</Text>
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>
    </Space>
  )
}

const fullHeightStyle = { height: '100%' }
const legendDotStyle = { width: 8, height: 8, borderRadius: '50%' }
const donutStyle = {
  width: 184,
  height: 184,
  maxWidth: '100%',
  margin: '0 auto',
  borderRadius: '50%',
  display: 'grid',
  placeItems: 'center',
  background: 'conic-gradient(#e6f4ff 0 58.2%, #1677ff 58.2% 70.8%, #d4b106 70.8% 79.6%, #d46b08 79.6% 90.7%, #cf1322 90.7% 100%)',
}
const donutCenterStyle = {
  width: 116,
  height: 116,
  borderRadius: '50%',
  background: '#fff',
  display: 'grid',
  placeItems: 'center',
  alignContent: 'center',
}
const trendChartStyle = {
  minHeight: 224,
  width: '100%',
  display: 'flex',
  alignItems: 'center',
  position: 'relative' as const,
  overflow: 'hidden',
}
const trendTooltipStyle = {
  position: 'absolute' as const,
  top: 8,
  right: 8,
  zIndex: 1,
  display: 'flex',
  flexDirection: 'column' as const,
  gap: 2,
  padding: '6px 10px',
  border: '1px solid #d9d9d9',
  borderRadius: 4,
  background: '#fff',
  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.12)',
}

export default RiskAnalyticsContent