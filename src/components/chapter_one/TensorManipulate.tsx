import { Box } from '@chakra-ui/react'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'
import pythonCode from '../../data/pythonCode'
import CodeDisplay from '../CodeDisplay'

const TensorManipulate = () => {
  const [ manipul, reshape, transpose, numpy, concat_stack ] = chapterOne[4].sections
  const [
    reshapeCode, transposeCode, numpyCode, concatStackCode 
  ] = pythonCode.slice(9, 13).map(obj => obj.code);

  return (
    <Box py={2}>
      <BaseGrid section={manipul}/>
      <BaseGrid section={reshape}/>
      <CodeDisplay codes={reshapeCode}/>
      <BaseGrid section={transpose}/>
      <CodeDisplay codes={transposeCode}/>
      <BaseGrid section={numpy}/>
      <CodeDisplay codes={numpyCode}/>
      <BaseGrid section={concat_stack}/>
      <CodeDisplay codes={concatStackCode}/>
    </Box>
  )
}

export default TensorManipulate