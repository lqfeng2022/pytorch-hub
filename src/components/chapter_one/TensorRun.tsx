import { Box, Text } from '@chakra-ui/react'
import chapterOne from '../../data/chapterOne'
import pythonCode from '../../data/pythonCode'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'

const TensorRun = () => {
  const [ 
    ongpu, rungpu, gpu, gpuf, cuda, cudaf, getgpu
  ] = chapterOne[7].sections
  const tensorrunCode = pythonCode[17].code
  
  return (
    <>
      <Box py={5}>
        <Text as='b' fontSize='lg' color='gray.600'>{ongpu.name}</Text>
        <Definition definition={ongpu}/>
      </Box>
      <BaseGrid section={rungpu}/>
      <CodeDisplay codes={tensorrunCode}/>
      <Box py={5}>
        <Text as='b' fontSize='lg' color='gray.600'>{gpu.name}</Text>
        <Definition definition={gpu}/>
      </Box>
      <BaseGrid section={gpuf}/>
      <LeftGrid section={cuda}/>
      <BaseGrid section={cudaf}/>
      <BaseGrid section={getgpu}/>
    </>
  )
}

export default TensorRun