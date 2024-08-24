import { Text } from '@chakra-ui/react'
import DLApps from './DLApps'
import DLComparing from './DLComparing'
import DLDefinition from './DLDefinition'
import DLRelationship from './DLRelationship'

const DLSection = () => {
  return (
    <>
      {/* 3. Deep Learning */}
      <Text as='b' fontSize='xl'>3. Deep Learning</Text>
      <DLDefinition/>
      <DLRelationship/>
      <DLComparing/>
      <DLApps/>
    </>
  )
}

export default DLSection