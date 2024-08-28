import { Box } from '@chakra-ui/react'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'

const TensorAttributs = () => {
  const [ attributes, shape ] = chapterOne[2].sections

  return (
    <Box py={2}>
      <BaseGrid section={attributes}/>
      <BaseGrid section={shape}/>
    </Box>
  )
}

export default TensorAttributs