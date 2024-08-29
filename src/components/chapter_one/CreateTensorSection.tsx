import { Box, Text } from '@chakra-ui/react'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import pythonCode from '../../data/pythonCode'

const CreateTensorSection = () => {
  const [ random, zerosOnes, range, likes ] =  chapterOne[1].sections
  const [
    randomCode, zerosCode, rangeCode, likesCode
  ] = pythonCode.slice(1, 5).map(obj => obj.code);

  return (
    <>
      <Box pt={5}>
        <Text as='b' fontSize='xl' color='gray.600'>2. Create TENSOR</Text>
      </Box>
      <BaseGrid section={random}/>
      <CodeDisplay codes={randomCode}/>
      <BaseGrid section={zerosOnes}/>
      <CodeDisplay codes={zerosCode}/>
      <BaseGrid section={range}/>
      <CodeDisplay codes={rangeCode}/>
      <BaseGrid section={likes}/>
      <CodeDisplay codes={likesCode}/>
    </>
  )
}

export default CreateTensorSection