import chapterFour from '../../data/chapterFour'
import Definition from '../Definition'
import codeBinaryModel from '../../data/codeBinaryModel'
import CodeDisplay from '../CodeDisplay'
import RightGrid from '../RightGrid'
import LeftGrid from '../LeftGrid'
import { Box } from '@chakra-ui/react'

const TrainModel = () => {
  const [ defin, train, test, loss ] = chapterFour[2].sections
  const [ trainCode, testCode, lossCode
  ] = codeBinaryModel.slice(4, 7).map(obj => obj.code)

  return (
    <Box pt={5}>
      <Definition title={defin.name} definition={defin}/>
      <RightGrid section={train}/>
      <CodeDisplay codes={trainCode}/>
      <LeftGrid section={test}/>
      <CodeDisplay codes={testCode}/>
      <RightGrid section={loss}/>
      <CodeDisplay codes={lossCode}/>
    </Box>
  )
}

export default TrainModel