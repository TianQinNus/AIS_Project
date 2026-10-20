export type VehicleTrendPoint = {
  label: string
  riskClass?: string
  risk6Percentile?: number | null
  risk48?: number | null
  risk24?: number | null
  risk12?: number | null
  risk6?: number | null
}

export type VehicleRiskReading = {
  label: string
  value: string
}

export type VehicleFactor = {
  name: string
  impact: number
  shapValue?: number | null
}

export type VehicleDetail = {
  vehicleId: string
  badge: string
  tagColor: string
  predictedState: string
  action: string
  history: string
  risks: VehicleRiskReading[]
  trend: VehicleTrendPoint[]
  rule: string
  reasons: string[]
  factors: VehicleFactor[]
}

const vehicleDetails: VehicleDetail[] = [
  {
    vehicleId: 'VHC-89321',
    badge: 'PRIORITY INSPECT',
    tagColor: 'red',
    predictedState: 'C4 · 0–6 units',
    action: 'Priority Inspect',
    history: 'Strong',
    risks: [
      { label: 'Risk48', value: '0.898' },
      { label: 'Risk24', value: '0.796' },
      { label: 'Risk12', value: '0.679' },
      { label: 'Risk6', value: '0.431' },
    ],
    trend: [
      { label: '48,481', risk48: 0.41, risk24: 0.28, risk12: 0.18, risk6: 0.08 },
      { label: '49,481', risk48: 0.55, risk24: 0.41, risk12: 0.31, risk6: 0.16 },
      { label: '50,481', risk48: 0.68, risk24: 0.55, risk12: 0.44, risk6: 0.26 },
      { label: '51,481', risk48: 0.81, risk24: 0.68, risk12: 0.57, risk6: 0.35 },
      { label: '52,481', risk48: 0.898, risk24: 0.796, risk12: 0.679, risk6: 0.431 },
    ],
    rule: 'TR1',
    reasons: ['C4', 'Risk6 top 5%', 'Persistent near-term risk', 'Fast deterioration'],
    factors: [
      { name: '397_12 rate', impact: 0.86 },
      { name: '167_3 change5', impact: 0.64 },
      { name: '044_8 delta3', impact: 0.41 },
      { name: '512_1 lag', impact: 0.28 },
    ],
  },
  {
    vehicleId: 'VHC-55217',
    badge: 'PRIORITY INSPECT',
    tagColor: 'red',
    predictedState: 'C4 · 0–6 units',
    action: 'Priority Inspect',
    history: 'Strong',
    risks: [
      { label: 'Risk48', value: '0.861' },
      { label: 'Risk24', value: '0.742' },
      { label: 'Risk12', value: '0.628' },
      { label: 'Risk6', value: '0.402' },
    ],
    trend: [
      { label: '44,102', risk48: 0.36, risk24: 0.24, risk12: 0.16, risk6: 0.07 },
      { label: '45,102', risk48: 0.49, risk24: 0.37, risk12: 0.28, risk6: 0.14 },
      { label: '46,102', risk48: 0.63, risk24: 0.51, risk12: 0.4, risk6: 0.23 },
      { label: '47,102', risk48: 0.75, risk24: 0.63, risk12: 0.52, risk6: 0.32 },
      { label: '48,102', risk48: 0.861, risk24: 0.742, risk12: 0.628, risk6: 0.402 },
    ],
    rule: 'TR1',
    reasons: ['C4', 'Risk6 top 5%', 'Persistent near-term risk', 'Fast deterioration'],
    factors: [
      { name: '381_6 rate', impact: 0.79 },
      { name: '204_2 change5', impact: 0.57 },
      { name: '119_4 delta3', impact: 0.36 },
      { name: '077_9 lag', impact: 0.22 },
    ],
  },
  {
    vehicleId: 'VHC-66107',
    badge: 'INSPECT',
    tagColor: 'orange',
    predictedState: 'C3 · 6–12 units',
    action: 'Plan Inspection',
    history: 'Moderate',
    risks: [
      { label: 'Risk48', value: '0.704' },
      { label: 'Risk24', value: '0.581' },
      { label: 'Risk12', value: '0.446' },
      { label: 'Risk6', value: '0.268' },
    ],
    trend: [
      { label: '32,890', risk48: 0.32, risk24: 0.22, risk12: 0.14, risk6: 0.06 },
      { label: '33,890', risk48: 0.41, risk24: 0.31, risk12: 0.21, risk6: 0.11 },
      { label: '34,890', risk48: 0.52, risk24: 0.4, risk12: 0.29, risk6: 0.16 },
      { label: '35,890', risk48: 0.61, risk24: 0.49, risk12: 0.37, risk6: 0.21 },
      { label: '36,890', risk48: 0.704, risk24: 0.581, risk12: 0.446, risk6: 0.268 },
    ],
    rule: 'TR2',
    reasons: ['C3', 'Risk12 above watch line', 'Rising operational risk', 'Moderate deterioration'],
    factors: [
      { name: '255_7 rate', impact: 0.58 },
      { name: '093_1 change5', impact: 0.44 },
      { name: '318_5 delta3', impact: 0.29 },
      { name: '140_2 lag', impact: 0.18 },
    ],
  },
  {
    vehicleId: 'VHC-77834',
    badge: 'INSPECT',
    tagColor: 'orange',
    predictedState: 'C3 · 6–12 units',
    action: 'Plan Inspection',
    history: 'Moderate',
    risks: [
      { label: 'Risk48', value: '0.662' },
      { label: 'Risk24', value: '0.547' },
      { label: 'Risk12', value: '0.413' },
      { label: 'Risk6', value: '0.241' },
    ],
    trend: [
      { label: '37,210', risk48: 0.29, risk24: 0.2, risk12: 0.13, risk6: 0.05 },
      { label: '38,210', risk48: 0.38, risk24: 0.28, risk12: 0.19, risk6: 0.1 },
      { label: '39,210', risk48: 0.48, risk24: 0.37, risk12: 0.27, risk6: 0.15 },
      { label: '40,210', risk48: 0.57, risk24: 0.46, risk12: 0.34, risk6: 0.19 },
      { label: '41,210', risk48: 0.662, risk24: 0.547, risk12: 0.413, risk6: 0.241 },
    ],
    rule: 'TR2',
    reasons: ['C3', 'Risk12 above watch line', 'Rising operational risk', 'Moderate deterioration'],
    factors: [
      { name: '176_4 rate', impact: 0.52 },
      { name: '408_9 change5', impact: 0.39 },
      { name: '062_3 delta3', impact: 0.25 },
      { name: '291_8 lag', impact: 0.16 },
    ],
  },
  {
    vehicleId: 'VHC-33120',
    badge: 'ENHANCED MONITOR',
    tagColor: 'gold',
    predictedState: 'C2 · 12–24 units',
    action: 'Monitor',
    history: 'Limited',
    risks: [
      { label: 'Risk48', value: '0.512' },
      { label: 'Risk24', value: '0.406' },
      { label: 'Risk12', value: '0.294' },
      { label: 'Risk6', value: '0.168' },
    ],
    trend: [
      { label: '25,340', risk48: 0.34, risk24: 0.26, risk12: 0.18, risk6: 0.09 },
      { label: '26,340', risk48: 0.38, risk24: 0.29, risk12: 0.21, risk6: 0.11 },
      { label: '27,340', risk48: 0.43, risk24: 0.33, risk12: 0.24, risk6: 0.13 },
      { label: '28,340', risk48: 0.47, risk24: 0.37, risk12: 0.27, risk6: 0.15 },
      { label: '29,340', risk48: 0.512, risk24: 0.406, risk12: 0.294, risk6: 0.168 },
    ],
    rule: 'TR3',
    reasons: ['C2', 'Risk6 outside top 5%', 'Risk holding across recent steps', 'Slow deterioration'],
    factors: [
      { name: '128_5 rate', impact: 0.37 },
      { name: '066_2 change5', impact: 0.29 },
      { name: '347_1 delta3', impact: 0.18 },
      { name: '019_7 lag', impact: 0.11 },
    ],
  },
]

export function getVehicleDetail(vehicleId: string) {
  return vehicleDetails.find((vehicle) => vehicle.vehicleId === vehicleId)
}
