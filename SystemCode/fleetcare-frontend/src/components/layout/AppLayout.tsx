import { Layout } from 'antd'
import { Outlet } from 'react-router-dom'
import FloatingAssistant from '../assistant/FloatingAssistant'
import AppHeader from './AppHeader'
import Sidebar from './Sidebar'

const { Content } = Layout

function AppLayout() {
  return (
    <Layout className="app-layout">
      <Sidebar />
      <Layout>
        <AppHeader />
        <Content className="app-content">
          <Outlet />
        </Content>
      </Layout>
      <FloatingAssistant />
    </Layout>
  )
}

export default AppLayout