import { extendTheme } from '@chakra-ui/react'
import { dividerTheme } from './components/Divider'

export const theme = extendTheme({
  components: { Divider: dividerTheme },
})