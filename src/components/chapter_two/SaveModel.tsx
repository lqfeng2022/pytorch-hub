import Definition from '../Definition'
import chapterTwo from '../../data/chapterTwo'
import codeLineModel from '../../data/codeLineModel'
import CodeDisplay from '../CodeDisplay'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'
import { Box } from '@chakra-ui/react'

const SaveModel = () => {
  const [ define, save, load ] = chapterTwo[4].sections
  const [ savecode, loadcode ] = codeLineModel.slice(9, 11).map(obj => obj.code)

  return (
    <Box pt={5}>
      <Definition title={define.name} definition={define}/>
      <LeftGrid section={save}/>
      <CodeDisplay codes={savecode}/>
      <RightGrid section={load}/>
      <CodeDisplay codes={loadcode}/>
    </Box>
  )
}

export default SaveModel