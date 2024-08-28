import { Box, Text } from '@chakra-ui/react'
import Definition from '../Definition'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'

const TensorSection = () => {
  const [ whats, how ] =  chapterOne[0].sections

  return (
    <>
      <Box py={5}>
        <Text as='b' fontSize='xl' color='gray.600'>1. TENSOR</Text>
        <Definition definition={whats}/>
      </Box>
      <BaseGrid section={how}/>
    </>
  )
}

export default TensorSection