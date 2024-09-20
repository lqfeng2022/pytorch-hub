import { Box } from '@chakra-ui/react'
import chapterFour from '../../data/chapterFour'
import codeBinaryModel from '../../data/codeBinaryModel'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'

const ImproveModel = () => {
  const [ defin, improve ] = chapterFour[3].sections
  const [ one, one_architec, one_build, one_visual, one_loss ] = chapterFour[4].sections
  const [ two, two_build, two_visual, two_loss ] = chapterFour[5].sections
  const [ three, three_build, three_visual, three_loss ] = chapterFour[6].sections

  const [ one_build_code, one_train_code, one_loss_code
  ] = codeBinaryModel.slice(7, 10).map(obj => obj.code)
  const [ two_build_code, two_train_code, two_loss_code
  ] = codeBinaryModel.slice(10, 13).map(obj => obj.code)
  const [ three_build_code, three_train_code, three_loss_code
  ] = codeBinaryModel.slice(13, 16).map(obj => obj.code)

  return (
    <Box pt={5}>
      <Definition title={defin.name} definition={defin}/>
      <BaseGrid section={improve}/>
      <div id='model_one'>
        <BaseGrid section={one}/>
        <LeftGrid section={one_architec}/>
        <RightGrid section={one_build}/>
        <CodeDisplay codes={one_build_code}/>
        <LeftGrid section={one_visual}/>
        <CodeDisplay codes={one_train_code}/>
        <RightGrid section={one_loss}/>
        <CodeDisplay codes={one_loss_code}/>
      </div>
      <div id='model_two'>
        <BaseGrid section={two}/>
        <RightGrid section={two_build}/>
        <CodeDisplay codes={two_build_code}/>
        <LeftGrid section={two_visual}/>
        <CodeDisplay codes={two_train_code}/>
        <RightGrid section={two_loss}/>
        <CodeDisplay codes={two_loss_code}/>
      </div>
      <div id='model_three'>
        <BaseGrid section={three}/>
        <RightGrid section={three_build}/>
        <CodeDisplay codes={three_build_code}/>
        <LeftGrid section={three_visual}/>
        <CodeDisplay codes={three_train_code}/>
        <RightGrid section={three_loss}/>
        <CodeDisplay codes={three_loss_code}/>
      </div>
    </Box>
  )
}

export default ImproveModel