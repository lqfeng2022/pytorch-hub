import { Box } from '@chakra-ui/react'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'
import pythonCode from '../../data/pythonCode'
import CodeDisplay from '../CodeDisplay'

const TensorOperations = () => {
  const [ 
    operations, addSub, matmul, mmWork, twoRule, mmWebs, dot,  dotTransf, aggre
  ] = chapterOne[3].sections
  const [ 
    addSubCode, matmulCode, aggreCode 
  ] = pythonCode.slice(6, 9).map(obj => obj.code)

  return (
    <Box py={2}>
      <BaseGrid section={operations}/>
      <BaseGrid section={addSub}/>
      <CodeDisplay codes={addSubCode}/>
      <BaseGrid section={matmul}/>
      <BaseGrid section={mmWork}/>
      <BaseGrid section={twoRule}/>
      <BaseGrid section={mmWebs}/>
      <BaseGrid section={dot}/>
      <BaseGrid section={dotTransf}/>
      <CodeDisplay codes={matmulCode}/>
      <BaseGrid section={aggre}/>
      <CodeDisplay codes={aggreCode}/>
    </Box>
  )
}

export default TensorOperations