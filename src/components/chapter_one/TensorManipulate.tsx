import { Box } from '@chakra-ui/react'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'

const TensorManipulate = () => {
  const [ manipul, aggregate, reshape, concat_stack ] = chapterOne[4].sections

  return (
    <Box py={2}>
      <BaseGrid section={manipul}/>
      <BaseGrid section={aggregate}/>
      <BaseGrid section={reshape}/>
      <BaseGrid section={concat_stack}/>
    </Box>
  )
}

export default TensorManipulate