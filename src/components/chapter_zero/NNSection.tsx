import { Box, Text } from '@chakra-ui/react'
import BaseGrid from '../BaseGrid'
import Definition from '../Definition'
import chapterOne from '../../data/chapterOne'

const NNSection = () => {
  const [ definition, model, neurons ] = chapterOne[3].sections
  
  return (
    <>
      <Box py={5}>
        <Text as='b' fontSize='xl' color='gray.600'>4. Neural Network</Text>
        <Definition definition={definition}/>
      </Box>
      <BaseGrid section={model}/>
      <BaseGrid section={neurons}/>
    </>
  )
}

export default NNSection