import chapterOne from '../../data/chapterOne'
import CodeDisplay from '../CodeDisplay'
import pythonCode from '../../data/codeTensors'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'
import { Box } from '@chakra-ui/react'

const TensorIndex = () => {
  const [ define, basic, slice ] = chapterOne[5].sections
  const [ indexbasicCode, indexsliceCode 
  ] = pythonCode.slice(18, 20).map(obj => obj.code)

  return (
    <Box pt={5}>
      <Definition title={define.name} definition={define}/>
      <LeftGrid section={basic}/>
      <CodeDisplay codes={indexbasicCode}/>
      <RightGrid section={slice}/>
      <CodeDisplay codes={indexsliceCode}/>
    </Box>
  )
}

export default TensorIndex