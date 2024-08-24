import { Text } from '@chakra-ui/react'
import Neurons from './Neurons'
import NNArchitecture from './NNArchitecture'
import NNDefinition from './NNDefinition'

const NNSection = () => {
  return (
    <>
      {/* 4. Neural Network */}
      <Text as='b' fontSize='xl'>4. Neural Network</Text>
      <NNDefinition/>
      <NNArchitecture/>
      <Neurons/>
    </>
  )
}

export default NNSection