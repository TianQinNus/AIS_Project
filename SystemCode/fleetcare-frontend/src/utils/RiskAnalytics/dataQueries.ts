export const riskDistribution = [
  { label: 'Continue (>48 steps)', count: '2,479', percentage: '58.2%', color: '#e6f4ff' },
  { label: 'Monitor (24-48 steps)', count: '536', percentage: '12.6%', color: '#1677ff' },
  { label: 'Enhanced Monitor (12-24 steps)', count: '374', percentage: '8.8%', color: '#d4b106' },
  { label: 'Inspect (6-12 steps)', count: '472', percentage: '11.1%', color: '#d46b08' },
  { label: 'Priority Inspect (0-6 steps)', count: '395', percentage: '9.3%', color: '#cf1322' },
]

export const componentRiskData = [
  { key: 'component-x', component: 'Component X', avgRisk: '68 /100', riskCount: 142, tagColor: 'orange' },
  { key: 'component-y', component: 'Component Y', avgRisk: '41 /100', riskCount: 58, tagColor: 'orange' },
  { key: 'component-z', component: 'Component Z', avgRisk: '27 /100', riskCount: 24, tagColor: 'blue' },
  { key: 'braking-system', component: 'Braking system', avgRisk: '55 /100', riskCount: 89, tagColor: 'orange' },
  { key: 'battery-pack', component: 'Battery pack', avgRisk: '33 /100', riskCount: 41, tagColor: 'orange' },
]

export const fleetSegments = [
  { label: 'Heavy trucks', count: '2,140', percent: 100, color: '#cf1322' },
  { label: 'Medium trucks', count: '1,320', percent: 60, color: '#d46b08' },
  { label: 'Light trucks', count: '796', percent: 30, color: '#d4b106' },
]

export const featureSignals = [
  { label: 'Temperature (f12)', status: 'High', color: '#cf1322' },
  { label: 'Vibration (f25)', status: 'Elevated', color: '#d46b08' },
  { label: 'Oil contamination (f44)', status: 'Elevated', color: '#d46b08' },
  { label: 'Pressure delta (f09)', status: 'Watch', color: '#d4b106' },
  { label: 'Load cycles (f31)', status: 'Normal', color: '#389e0d' },
]

export const riskTrendData = [
  { date: 'Sep 6', value: 82 },
  { date: 'Sep 7', value: 76 },
  { date: 'Sep 8', value: 79 },
  { date: 'Sep 9', value: 64 },
  { date: 'Sep 10', value: 70 },
  { date: 'Sep 11', value: 55 },
  { date: 'Sep 12', value: 61 },
]