import { Box, Stack, Text } from '@chakra-ui/react'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'

const TensorIndex = () => {
  const [ basic, slice ] = chapterOne[6].sections

  return (
    <Box py={5}>
      <Stack>
        <Text as='b' fontSize='xl' color='gray.600'>{chapterOne[6].name}</Text>
        <Text as='i'>{chapterOne[6].description}</Text>
      </Stack>
      <BaseGrid section={basic}/>
      <BaseGrid section={slice}/>
    </Box>
  )
}

export default TensorIndex