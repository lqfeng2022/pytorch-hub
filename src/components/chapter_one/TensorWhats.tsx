import Definition from '../Definition'
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import pythonCode from '../../data/codeTensors'
import { Box } from '@chakra-ui/react'

const TensorWhats = () => {
  const [ whats, how ] =  chapterOne[0].sections
  const tensorCode = pythonCode[0].code

  return (
    <Box pt={5}>
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={how}/>
      <CodeDisplay codes={tensorCode}/>
    </Box>
  )
}

export default TensorWhats