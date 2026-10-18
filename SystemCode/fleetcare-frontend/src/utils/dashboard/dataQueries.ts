import { dashboardEndpoints } from './endpoints'
import { API_BASE_URL } from '../../config/api'

export type DashboardResponse = {
	fleet: { total: number; analyzed: number }
	risk: {
		at_risk: number
		high: number
		critical: number
		average_score: number
		previous_average_score?: number
		average_score_change?: number
		average_score_trend?: 'UP' | 'DOWN' | 'STABLE'
	}
	risk_snapshot: Array<{ action: string; count: number; percentage: number }>
	maintenance_plan: { immediate: number; short_term: number; monitor: number; manual_review: number }
	top_risk_alerts: Array<{
		analysis_id: number
		vehicle_id: number
		state: string
		risk_score: number
		priority: string
		action: string
		reason: string
	}>
}

export async function fetchDashboard(signal?: AbortSignal): Promise<DashboardResponse> {
	const response = await fetch(`${API_BASE_URL}${dashboardEndpoints.overview}`, {
		headers: { Accept: 'application/json' },
		cache: 'no-store',
		signal,
	})

	if (!response.ok) {
		throw new Error(`Dashboard request failed (${response.status})`)
	}

	return response.json() as Promise<DashboardResponse>
}
