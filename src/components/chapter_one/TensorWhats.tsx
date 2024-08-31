import { Box, Text } from '@chakra-ui/react'
import Definition from '../Definition'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import pythonCode from '../../data/pythonCode'

const TensorWhats = () => {
  const [ whats, how ] =  chapterOne[0].sections
  const tensorCode = pythonCode[0].code

  return (
    <>
      <Box py={5}>
        <Text as='b' fontSize='xl' color='gray.600'>1. What's a Tensor</Text>
        <Definition definition={whats}/>
      </Box>
      <BaseGrid section={how}/>
      <CodeDisplay codes={tensorCode}/>
    </>
  )
}

export default TensorWhats