import { Box } from '@chakra-ui/react'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'
import pythonCode from '../../data/codeTensors'
import CodeDisplay from '../CodeDisplay'
import LeftGrid from '../LeftGrid'

const TensorManipulate = () => {
  const [ manipul, changeshape, transpose, permute, reshape, numpy, 
    concat_stack, stack, vstack, hstack
  ] = chapterOne[4].sections
  const [ squeezeCode, transposeCode, permuteCode, reshapeCode, numpyCode, 
    concatStackCode, stackCode, vstackCode, hstackCode
  ] = pythonCode.slice(9, 18).map(obj => obj.code);

  return (
    <Box pt={5}>
      <BaseGrid section={manipul}/>
      <BaseGrid section={changeshape}/>
      <CodeDisplay codes={squeezeCode}/>
      <LeftGrid section={transpose}/>
      <CodeDisplay codes={transposeCode}/>
      <BaseGrid section={permute}/>
      <CodeDisplay codes={permuteCode}/>
      <BaseGrid section={reshape}/>
      <CodeDisplay codes={reshapeCode}/>
      <BaseGrid section={numpy}/>
      <CodeDisplay codes={numpyCode}/>
      <LeftGrid section={concat_stack}/>
      <CodeDisplay codes={concatStackCode}/>
      <BaseGrid section={stack}/>
      <CodeDisplay codes={stackCode}/>
      <BaseGrid section={vstack}/>
      <CodeDisplay codes={vstackCode}/>
      <BaseGrid section={hstack}/>
      <CodeDisplay codes={hstackCode}/>
    </Box>
  )
}

export default TensorManipulate