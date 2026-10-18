import { Navigate, createBrowserRouter } from 'react-router-dom'
import AppLayout from '../components/layout/AppLayout'
import Assistant from '../pages/Assistant'
import Dashboard from '../pages/Dashboard'
import NotFound from '../pages/NotFound'
import RiskAnalytics from '../pages/RiskAnalytics'
import Vehicles from '../pages/Vehicles'
import VehicleDetail from '../pages/Vehicles/Detail'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      { index: true, element: <Navigate to="/dashboard" replace /> },
      { path: 'dashboard', element: <Dashboard /> },
      { path: 'vehicles', element: <Vehicles /> },
      { path: 'vehicles/:vehicleId', element: <VehicleDetail /> },
      { path: 'risk-analytics', element: <RiskAnalytics /> },
      { path: 'assistant', element: <Assistant /> },
      { path: '*', element: <NotFound /> },
    ],
  },
])