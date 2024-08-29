import { Box, Stack, Text } from '@chakra-ui/react'
import Definition from '../Definition'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'
import LeftGrid from '../LeftGrid'

const TensorRun = () => {
  const [ 
    rungpu, gpu, gpuf, cuda, cudaf, getgpu, howrungpu
  ] = chapterOne[9].sections
  
  return (
    <>
      <Box py={5}>
        <Text as='b' fontSize='xl' color='gray.600'>{rungpu.name}</Text>
        <Definition definition={rungpu}/>
      </Box>
      <Box py={5}>
        <Text as='b' color='gray.600'>{gpu.name}</Text>
        <Definition definition={gpu}/>
      </Box>
      <BaseGrid section={gpuf}/>
      <LeftGrid section={cuda}/>
      <BaseGrid section={cudaf}/>
      <BaseGrid section={getgpu}/>
      <BaseGrid section={howrungpu}/>
    </>
  )
}

export default TensorRun