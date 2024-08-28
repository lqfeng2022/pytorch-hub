import { Box } from '@chakra-ui/react'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'

const TensorOperations = () => {
  const [ 
    operations, addSub, mulDiv, matmul, dot, dotRule, dotWork, dotApp, dotWebs
  ] = chapterOne[3].sections

  return (
    <Box py={2}>
      <BaseGrid section={operations}/>
      <BaseGrid section={addSub}/>
      <BaseGrid section={mulDiv}/>
      <BaseGrid section={matmul}/>
      <BaseGrid section={dot}/>
      <BaseGrid section={dotRule}/>
      <BaseGrid section={dotWork}/>
      <BaseGrid section={dotApp}/>
      <BaseGrid section={dotWebs}/>
    </Box>
  )
}

export default TensorOperations