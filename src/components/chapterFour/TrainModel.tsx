import chapterFour from '../../data/chapterFour'
import Definition from '../Definition'
import BaseGrid from '../BaseGrid'
import codeBinaryModel from '../../data/codeBinaryModel'
import CodeDisplay from '../CodeDisplay'

const TrainModel = () => {
  const [ defin, train, test, loss ] = chapterFour[2].sections
  const [ 
    trainCode, testCode, lossCode
  ] = codeBinaryModel.slice(4, 7).map(obj => obj.code)

  return (
    <div>
      <Definition title={defin.name} definition={defin}/>
      <BaseGrid section={train}/>
      <CodeDisplay codes={trainCode}/>
      <BaseGrid section={test}/>
      <CodeDisplay codes={testCode}/>
      <BaseGrid section={loss}/>
      <CodeDisplay codes={lossCode}/>
    </div>
  )
}

export default TrainModel