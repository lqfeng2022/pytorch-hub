import chapterFour from '../../data/chapterFour'
import codeBinaryModel from '../../data/codeBinaryModel'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import Definition from '../Definition'
import RightGrid from '../RightGrid'

const BuildModel = () => {
  const [ defin, model, architect ] = chapterFour[1].sections
  const buildCode = codeBinaryModel[3].code

  return (
    <div>
      <Definition title={defin.name} definition={defin}/>
      <BaseGrid section={model}/>
      <CodeDisplay codes={buildCode}/>
      <RightGrid section={architect}/>
    </div>
  )
}

export default BuildModel