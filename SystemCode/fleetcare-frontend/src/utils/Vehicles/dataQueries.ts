export type VehicleAction = {
  action: string
  label: string
  count: number
}

export type VehicleActionListResponse = {
  items: VehicleAction[]
}

export type VehicleRecord = {
  vehicleId: string
  state: string
  predictedClass: string
  riskScore: number
  inMaintenance: boolean
}

export type VehicleListResponse = {
  total: number
  page: number
  pageSize: number
  items: VehicleRecord[]
}
