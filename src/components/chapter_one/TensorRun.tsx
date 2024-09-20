import { Box } from '@chakra-ui/react'
import chapterOne from '../../data/chapterOne'
import pythonCode from '../../data/codeTensors'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'

const TensorRun = () => {
  const [ ongpu, rungpu, gpu, gpuf, cuda, cudaf, getgpu
  ] = chapterOne[7].sections
  const tensorrunCode = pythonCode[22].code
  
  return (
    <Box pt={5}>
      <Definition title={ongpu.name} definition={ongpu}/>
      <BaseGrid section={rungpu}/>
      <CodeDisplay codes={tensorrunCode}/>
      <Definition title={gpu.name} definition={gpu}/>
      <BaseGrid section={gpuf}/>
      <LeftGrid section={cuda}/>
      <BaseGrid section={cudaf}/>
      <BaseGrid section={getgpu}/>
    </Box>
  )
}

export default TensorRun