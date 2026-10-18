import { useParams } from 'react-router-dom'
import VehicleDetailContent from '../../components/vehicles/VehicleDetailContent'

function VehicleDetail() {
  const { vehicleId = '' } = useParams()

  return <VehicleDetailContent vehicleId={decodeURIComponent(vehicleId)} />
}

export default VehicleDetail
