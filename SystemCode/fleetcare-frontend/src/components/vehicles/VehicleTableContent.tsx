import { SearchOutlined } from '@ant-design/icons'
import {
  Alert,
  Button,
  Card,
  Flex,
  Input,
  Pagination,
  Select,
  Space,
  Switch,
  Table,
  Tag,
  Typography,
} from 'antd'
import type { TableColumnsType } from 'antd'
import { useEffect, useState } from 'react'
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { getJson, patchJson } from '../../utils/http'
import { type VehicleAction, type VehicleListResponse, type VehicleRecord } from '../../utils/Vehicles/dataQueries'
import { vehicleEndpoints } from '../../utils/Vehicles/endpoints'

const { Text, Title } = Typography

const urgencyColors = ['#cf1322', '#d46b08', '#d4b106', '#1677ff', '#389e0d', '#722ed1', '#8c8c8c']
const stateColors: Record<string, string> = {
  C0: 'green',
  C1: 'blue',
  C2: 'gold',
  C3: 'orange',
  C4: 'red',
}
const pageSize = 10

function VehicleTableContent() {
  const navigate = useNavigate()
  const location = useLocation()
  const [searchParams, setSearchParams] = useSearchParams()
  const page = Math.max(1, Number(searchParams.get('page')) || 1)
  const searchTerm = searchParams.get('q') ?? ''
  const selectedStatus = searchParams.get('class') || 'All classes'
  const [searchInput, setSearchInput] = useState(searchTerm)
  const [vehicles, setVehicles] = useState<VehicleRecord[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [errorMessage, setErrorMessage] = useState('')
  const [savingId, setSavingId] = useState<string | null>(null)
  const [actions, setActions] = useState<VehicleAction[]>([])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const next = searchInput.trim()
      if (next === searchTerm) return
      setSearchParams((current) => {
        const params = new URLSearchParams(current)
        if (next) params.set('q', next)
        else params.delete('q')
        params.delete('page')
        return params
      }, { replace: true })
    }, 300)
    return () => window.clearTimeout(timer)
  }, [searchInput, searchTerm, setSearchParams])

  const openVehicle = (vehicleId: string) => {
    navigate(`/vehicles/${vehicleId}`, {
      state: { from: `${location.pathname}${location.search}` },
    })
  }

  const updatePage = (nextPage: number) => {
    setSearchParams((current) => {
      const params = new URLSearchParams(current)
      if (nextPage <= 1) params.delete('page')
      else params.set('page', String(nextPage))
      return params
    }, { replace: true })
  }

  useEffect(() => {
    let cancelled = false
    getJson<{ items: VehicleAction[] }>(vehicleEndpoints.actions)
      .then((data) => {
        if (!cancelled) setActions(data.items)
      })
      .catch(() => {
        if (!cancelled) setActions([])
      })
    return () => {
      cancelled = true
    }
  }, [])

  const selectedAction = actions.find(
    (item) => item.action === selectedStatus || item.label === selectedStatus,
  )

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setErrorMessage('')

    getJson<VehicleListResponse>(vehicleEndpoints.list, {
      q: searchTerm,
      predicted_class: selectedStatus === 'All classes' ? undefined : (selectedAction?.action ?? selectedStatus),
      page,
      page_size: pageSize,
    })
      .then((data) => {
        if (cancelled) return
        setVehicles(data.items)
        setTotal(data.total)
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setVehicles([])
        setTotal(0)
        setErrorMessage(error instanceof Error ? error.message : 'Unable to load vehicles')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [searchTerm, selectedStatus, selectedAction, page])

  const columns: TableColumnsType<VehicleRecord> = [
    {
      title: 'Vehicle ID',
      dataIndex: 'vehicleId',
      key: 'vehicleId',
      width: '20%',
      render: (vehicleId: string) => (
        <Typography.Link onClick={() => openVehicle(vehicleId)}>
          {vehicleId}
        </Typography.Link>
      ),
    },
    {
      title: 'Predicted Class',
      dataIndex: 'state',
      key: 'state',
      width: '20%',
      render: (state: string) => <Tag color={stateColors[state] ?? 'default'}>{state}</Tag>,
    },
    {
      title: 'Risk Score',
      dataIndex: 'riskScore',
      key: 'riskScore',
      width: '20%',
      render: (score: number) => <Text strong>{score} /100</Text>,
    },
    {
      title: 'Action',
      key: 'action',
      width: '20%',
      render: (_, vehicle) => {
        const rank = actions.findIndex((item) => item.label === vehicle.predictedClass)
        const background = rank >= 0 ? urgencyColors[rank] ?? urgencyColors[urgencyColors.length - 1] : undefined
        return (
          <Button
            size="small"
            type="primary"
            style={background ? { background, borderColor: background, color: '#fff' } : undefined}
            onClick={() => openVehicle(vehicle.vehicleId)}
          >
            {vehicle.predictedClass}
          </Button>
        )
      },
    },
    {
      title: 'Maintenance',
      dataIndex: 'inMaintenance',
      key: 'inMaintenance',
      width: '20%',
      render: (inMaintenance: boolean, vehicle) => (
        <Switch
          checked={inMaintenance}
          checkedChildren="On"
          unCheckedChildren="Off"
          loading={savingId === vehicle.vehicleId}
          onChange={(checked) => {
            const previous = vehicle.inMaintenance
            setVehicles((current) =>
              current.map((item) =>
                item.vehicleId === vehicle.vehicleId ? { ...item, inMaintenance: checked } : item,
              ),
            )
            setSavingId(vehicle.vehicleId)
            patchJson<{ inMaintenance: boolean }>(vehicleEndpoints.maintenance(vehicle.vehicleId), {
              inMaintenance: checked,
            })
              .catch((error: unknown) => {
                setVehicles((current) =>
                  current.map((item) =>
                    item.vehicleId === vehicle.vehicleId ? { ...item, inMaintenance: previous } : item,
                  ),
                )
                setErrorMessage(error instanceof Error ? error.message : 'Unable to update maintenance')
              })
              .finally(() => setSavingId(null))
          }}
        />
      ),
    },
  ]

  const from = total === 0 ? 0 : (page - 1) * pageSize + 1
  const to = Math.min(page * pageSize, total)

  return (
    <Card
      title={<Title level={5} style={{ margin: 0 }}>VEHICLE RISK OVERVIEW</Title>}
      extra={
        <Space wrap>
          <Input.Search
            allowClear
            value={searchInput}
            placeholder="Search vehicle ID..."
            prefix={<SearchOutlined />}
            onChange={(event) => setSearchInput(event.target.value)}
          />
          <Select
            value={selectedAction?.action ?? selectedStatus}
            style={{ minWidth: 180 }}
            options={[
              { value: 'All classes', label: 'All classes' },
              ...actions.map((item) => ({
                value: item.action,
                label: item.label,
              })),
            ]}
            onChange={(value) => {
              setSearchParams((current) => {
                const params = new URLSearchParams(current)
                if (value === 'All classes') params.delete('class')
                else params.set('class', value)
                params.delete('page')
                return params
              }, { replace: true })
            }}
          />
        </Space>
      }
    >
      {errorMessage && (
        <Alert type="error" showIcon title="Unable to load vehicles" description={errorMessage} style={{ marginBottom: 16 }} />
      )}
      <Table<VehicleRecord>
        rowKey="vehicleId"
        columns={columns}
        dataSource={vehicles}
        loading={loading}
        pagination={false}
        size="small"
        tableLayout="fixed"
      />
      <Flex justify="flex-end" align="center" gap={16} wrap style={paginationStyle}>
        <Text type="secondary">Showing {from} to {to} of {total.toLocaleString()} vehicles</Text>
        <Pagination
          current={page}
          pageSize={pageSize}
          total={total}
          showSizeChanger={false}
          onChange={updatePage}
        />
      </Flex>
    </Card>
  )
}

const paginationStyle = { marginTop: 24 }

export default VehicleTableContent
