import { Box, Stack, Text } from '@chakra-ui/react'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import pythonCode from '../../data/pythonCode'

const TensorIndex = () => {
  const [ basic, slice ] = chapterOne[5].sections
  const [
    indexbasicCode, indexsliceCode 
  ] = pythonCode.slice(13, 15).map(obj => obj.code)

  return (
    <Box py={5}>
      <Stack>
        <Text as='b' fontSize='lg' color='gray.600'>{chapterOne[5].name}</Text>
        <Text as='i'>{chapterOne[5].description}</Text>
      </Stack>
      <BaseGrid section={basic}/>
      <CodeDisplay codes={indexbasicCode}/>
      <BaseGrid section={slice}/>
      <CodeDisplay codes={indexsliceCode}/>
    </Box>
  )
}

export default TensorIndex