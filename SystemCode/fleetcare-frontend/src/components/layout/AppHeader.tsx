import { BellOutlined, UserOutlined } from '@ant-design/icons'
import { Avatar, Layout, Space, Typography } from 'antd'
import { useLocation } from 'react-router-dom'

const { Header } = Layout

function AppHeader() {
  const { pathname } = useLocation()
  const pageTitles: Record<string, string> = {
    '/dashboard': 'Dashboard',
    '/vehicles': 'Vehicles',
    '/risk-analytics': 'Risk Analytics',
    '/assistant': 'AI Assistant',
  }

  return (
    <Header className="app-header">
      <Typography.Title level={4}>
        {pageTitles[pathname] ?? (pathname.startsWith('/vehicles/') ? 'Vehicles' : 'Not Found')}
      </Typography.Title>
      <Space size="middle">
        <BellOutlined className="header-action" />
        <Typography.Text>Fleet Manager</Typography.Text>
        <Avatar icon={<UserOutlined />} />
      </Space>
    </Header>
  )
}

export default AppHeader