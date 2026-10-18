import { Layout, Menu, Typography } from 'antd'
import { useLocation, useNavigate } from 'react-router-dom'
import { navigationItems } from '../../config/navigation'

const { Sider } = Layout

function Sidebar() {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <Sider collapsible width={240} theme="dark">
      <div className="sidebar-brand">
        <Typography.Title level={4}>FleetCare AI</Typography.Title>
      </div>
      <Menu
        theme="dark"
        mode="inline"
        items={navigationItems}
        selectedKeys={[location.pathname.startsWith('/vehicles') ? '/vehicles' : location.pathname]}
        onClick={({ key }) => navigate(key)}
      />
    </Sider>
  )
}

export default Sidebar