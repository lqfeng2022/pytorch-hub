import chapterTwo from '../../data/chapterTwo'
import codeLineModel from '../../data/codeLineModel'
import CodeDisplay from '../CodeDisplay'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'

const PrepareData = () => {
  const [ prepare, create, split, visual ] = chapterTwo[1].sections
  const [
    createCode, splitCode, visualCode
  ] = codeLineModel.slice(0, 3).map(obj => obj.code);
  
  return (
    <div>
      <Definition title={prepare.name} definition={prepare}/>
      <LeftGrid section={create}/>
      <CodeDisplay codes={createCode}/>
      <RightGrid section={split}/>
      <CodeDisplay codes={splitCode}/>
      <LeftGrid section={visual}/>
      <CodeDisplay codes={visualCode}/>
    </div>
  )
}

export default PrepareData