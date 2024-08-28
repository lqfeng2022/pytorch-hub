import { Box, Text } from '@chakra-ui/react'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'

const CreateTensorSection = () => {
  const [ random, zerosOnes, range, likes ] =  chapterOne[1].sections

  return (
    <>
      <Box pt={5}>
        <Text as='b' fontSize='xl' color='gray.600'>2. Create TENSOR</Text>
      </Box>
      <BaseGrid section={random}/>
      <BaseGrid section={zerosOnes}/>
      <BaseGrid section={range}/>
      <BaseGrid section={likes}/>
    </>
  )
}

export default CreateTensorSection