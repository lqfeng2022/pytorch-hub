import Definition from '../Definition'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'
import chapterTwo from '../../data/chapterTwo'
import codeLineModel from '../../data/codeLineModel'
import CodeDisplay from '../CodeDisplay'

const TrainModel = () => {
  const [ 
    define, loop, visual_before, visual_train, visual_test, loss_curves 
  ] = chapterTwo[3].sections
  const [ 
    loopCode, visualCode, testCode, losscurvesCode 
  ] = codeLineModel.slice(5, 9).map(obj => obj.code)

  return (
    <div>
      <Definition title={define.name} definition={define}/>
      <RightGrid section={loop}/>
      <CodeDisplay codes={loopCode}/>
      <LeftGrid section={visual_before}/>
      <RightGrid section={visual_train}/>
      <CodeDisplay codes={visualCode}/>
      <LeftGrid section={visual_test}/>
      <CodeDisplay codes={testCode}/>
      <RightGrid section={loss_curves}/>
      <CodeDisplay codes={losscurvesCode}/>
    </div>
  )
}

export default TrainModel