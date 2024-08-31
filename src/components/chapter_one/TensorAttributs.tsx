import { Box } from '@chakra-ui/react'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import pythonCode from '../../data/pythonCode'

const TensorAttributs = () => {
  const [ attributes, shape ] = chapterOne[2].sections
  const attributesCode = pythonCode[5].code

  return (
    <>
      <Box py={2}>
        <BaseGrid section={attributes}/>
        <BaseGrid section={shape}/>
      </Box>
      <CodeDisplay codes={attributesCode}/>
    </>
  )
}

export default TensorAttributs