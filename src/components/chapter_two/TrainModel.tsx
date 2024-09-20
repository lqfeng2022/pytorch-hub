import Definition from '../Definition'
import chapterTwo from '../../data/chapterTwo'
import codeLineModel from '../../data/codeLineModel'
import CodeDisplay from '../CodeDisplay'
import BaseGrid from '../BaseGrid'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'
import { Box } from '@chakra-ui/react'

const TrainModel = () => {
  const [ define, loop, visual_before, visual_train, visual_test, loss_curves 
  ] = chapterTwo[3].sections
  const [ loopCode, visualCode, testCode, losscurvesCode 
  ] = codeLineModel.slice(5, 9).map(obj => obj.code)

  return (
    <Box pt={5}>
      <Definition title={define.name} definition={define}/>
      <BaseGrid section={loop}/>
      <CodeDisplay codes={loopCode}/>
      <LeftGrid section={visual_before}/>
      <RightGrid section={visual_train}/>
      <CodeDisplay codes={visualCode}/>
      <BaseGrid section={visual_test}/>
      <CodeDisplay codes={testCode}/>
      <BaseGrid section={loss_curves}/>
      <CodeDisplay codes={losscurvesCode}/>
    </Box>
  )
}

export default TrainModel