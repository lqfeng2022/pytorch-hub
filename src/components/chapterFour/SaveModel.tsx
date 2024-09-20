import Definition from '../Definition'
import CodeDisplay from '../CodeDisplay'
import BaseGrid from '../BaseGrid'
import chapterFour from '../../data/chapterFour'
import codeBinaryModel from '../../data/codeBinaryModel'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'
import { Box } from '@chakra-ui/react'

const SaveModel = () => {
  const [ define, choose, save, load ] = chapterFour[7].sections
  const [ code_choose, code_save, code_load 
  ] = codeBinaryModel.slice(16, 19).map(obj => obj.code)

  return (
    <Box pt={5}>
      <Definition title={define.name} definition={define}/>
      <BaseGrid section={choose}/>
      <CodeDisplay codes={code_choose}/>
      <LeftGrid section={save}/>
      <CodeDisplay codes={code_save}/>
      <RightGrid section={load}/>
      <CodeDisplay codes={code_load}/>
    </Box>
  )
}

export default SaveModel